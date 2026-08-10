"""LeRobot adapter for six-axis JAKA robots.

The public interface is deliberately small: construct :class:`JakaRobot`,
call the standard LeRobot lifecycle methods, and use joint or Cartesian
actions.  Vendor-specific operations remain available through ``robot.rc``
after connection instead of being mirrored as a second, partial SDK.

JAKA uses millimetres and radians. This adapter exposes metres and radians at
its LeRobot interface, converting only at the SDK seam.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .pose_math import matrix_to_rpy

logger = logging.getLogger(__name__)


# JAKA SDK constants used by the adapter. The vendor documentation presents
# these as numeric literals, so naming them here prevents those literals from
# leaking into callers.
ABS, INCR, CONT = 0, 1, 2
COORD_BASE, COORD_JOINT, COORD_TOOL = 0, 1, 2
IO_CABINET, IO_TOOL, IO_EXTEND = 0, 1, 2
PLANNER_DISABLED, PLANNER_T, PLANNER_S = -1, 0, 1
SERVO_CYCLE_S = 0.008
SERVO_SPIN_THRESHOLD_S = 0.0005
DEFAULT_SERVO_STEP_NUM = 1
SERVO_QUEUE_MAX = 100

_ERROR_MESSAGES = {
    -3: "communication failure or controller is unavailable",
    -6: "robot is not powered on",
    -7: "robot is not enabled",
    -58: "EDG is active; regular Servo Move is unavailable",
    -62: "Servo Move queue is full",
    -63: "Servo Move has not been enabled",
}


class JakaError(RuntimeError):
    """A non-zero return code from the JAKA SDK."""

    def __init__(self, operation: str, code: int, payload: tuple[Any, ...] = ()):
        self.operation = operation
        self.code = int(code)
        self.payload = payload
        detail = _ERROR_MESSAGES.get(self.code, "unknown SDK error")
        super().__init__(f"JAKA {operation} failed ({self.code}): {detail}")


def _sdk_directory() -> Path:
    return Path(__file__).resolve().parent


def _load_jkrc() -> Any:
    """Load the vendored native SDK only when a controller is used.

    ``libjakaAPI.so`` must be loaded globally before ``jkrc.so``. Doing this
    lazily keeps importing the robot configuration independent of native
    libraries and makes fake-SDK tests straightforward.
    """

    sdk_dir = _sdk_directory()
    api_library = sdk_dir / "libjakaAPI.so"
    if not api_library.is_file():
        raise ImportError(f"JAKA SDK library is missing: {api_library}")
    try:
        ctypes.CDLL(str(api_library), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise ImportError(f"Unable to load JAKA SDK library {api_library}: {exc}") from exc

    sdk_path = str(sdk_dir)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    try:
        return importlib.import_module("jkrc")
    except ImportError as exc:
        raise ImportError(f"Unable to import JAKA Python module from {sdk_dir}: {exc}") from exc


def create_rc(ip: str) -> Any:
    """Create a controller handle. Kept as an internal seam for tests."""

    return _load_jkrc().RC(ip)


def _payload(operation: str, result: Any) -> tuple[Any, ...]:
    """Validate a JAKA return tuple and return its flattened payload."""

    if not isinstance(result, tuple) or not result:
        raise JakaError(operation, -1, (result,))
    try:
        code = int(result[0])
    except (TypeError, ValueError) as exc:
        raise JakaError(operation, -1, tuple(result[1:])) from exc

    payload = tuple(result[1:])
    if len(payload) == 1 and isinstance(payload[0], (list, tuple)):
        payload = tuple(payload[0])
    if code:
        raise JakaError(operation, code, payload)
    return payload


def _vector(values: Any, *, name: str) -> np.ndarray:
    """Return one finite six-element vector, or raise a caller-facing error."""

    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain six numeric values") from exc
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain six finite numeric values")
    return vector


def _approach_vector(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    """Move a vector toward a target by at most one Euclidean step."""

    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= max_delta or distance == 0.0:
        return target.copy()
    return current + delta * (max_delta / distance)


def _wait_for_servo_deadline(
    deadline: float,
    *,
    stop: threading.Event,
    clock: Callable[[], float] = time.monotonic,
    spin_threshold_s: float = SERVO_SPIN_THRESHOLD_S,
) -> bool:
    """Wait until one Servo deadline; return false when shutdown is requested."""

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return not stop.is_set()
        if remaining > spin_threshold_s:
            if stop.wait(remaining - spin_threshold_s):
                return False
            continue
        while clock() < deadline:
            if stop.is_set():
                return False
        return not stop.is_set()


def _euler_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """Convert XYZ roll/pitch/yaw radians to a normalized wxyz quaternion."""

    roll, pitch, yaw = rpy / 2.0
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_to_euler(quaternion: np.ndarray, *, reference_rpy: np.ndarray | None = None) -> np.ndarray:
    """Convert normalized wxyz quaternion to continuous XYZ roll/pitch/yaw radians."""

    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    matrix = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    return matrix_to_rpy(matrix, reference_rpy=reference_rpy)


def _quaternion_distance(start: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the shortest-arc target quaternion and angular distance."""

    dot = float(np.dot(start, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    return target, 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _quaternion_slerp(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two wxyz quaternions along their shortest arc."""

    target, angle = _quaternion_distance(start, target)
    if angle < 1e-9:
        return target.copy()
    half_angle = angle / 2.0
    sin_half_angle = math.sin(half_angle)
    result = (
        math.sin((1.0 - fraction) * half_angle) / sin_half_angle * start
        + math.sin(fraction * half_angle) / sin_half_angle * target
    )
    return result / np.linalg.norm(result)


@RobotConfig.register_subclass("jaka_robot")
@dataclass
class JakaRobotConfig(RobotConfig):
    """Connection, control, and optional IO configuration for :class:`JakaRobot`."""

    ip: str = "192.168.1.31"
    type: str = "jaka_robot"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    auto_power_on: bool = True
    auto_enable: bool = True
    auto_enable_servo: bool = True
    # Preserve the controller's servo-enable state across SDK reconnects by
    # default. Continuous Servo Move is still always exited on disconnect.
    disable_on_disconnect: bool = False
    power_off_on_disconnect: bool = False
    use_grpc: bool = False
    # Open a second SDK handle for feedback/auxiliary IO so state reads cannot
    # interrupt the 8 ms Servo P command stream. Recorder workloads enable this.
    separate_feedback_connection: bool = False
    tool_id: int = 0
    user_frame_id: int = 0
    collision_level: int = 3

    # Planner-backed point moves, used only by ``move_j`` and ``move_l``.
    joint_speed: float = 0.5
    linear_speed_mm_s: float = 100.0

    # Safety limits applied to continuous LeRobot actions before Servo Move.
    max_relative_target: float | dict[str, float] | None = 0.05
    max_eef_step_m: float = 0.01
    max_eef_step_rad: float = 0.08
    joint_position_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    eef_pose_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    servo_step_num: int = DEFAULT_SERVO_STEP_NUM
    servo_queue_warn_depth: int = 80
    servo_joint_max_velocity_rad_s: float = 1.0
    servo_joint_max_acceleration_rad_s2: float = 2.0
    servo_eef_max_velocity_m_s: float = 0.05
    servo_eef_max_acceleration_m_s2: float = 0.2
    servo_eef_max_angular_velocity_rad_s: float = 0.5
    servo_eef_max_angular_acceleration_rad_s2: float = 1.0
    # Optional controller-side Servo Move jitter suppression. ``none`` preserves the
    # historical behavior; Cartesian NLF is the appropriate filter for Servo P targets.
    servo_filter_mode: Literal["none", "joint_lpf", "joint_nlf", "cartesian_nlf"] = "none"
    servo_filter_joint_lpf_cutoff_hz: float = 5.0
    servo_filter_joint_max_jerk_rad_s3: float = 10.0
    servo_filter_eef_max_jerk_m_s3: float = 1.0
    servo_filter_eef_max_angular_jerk_rad_s3: float = 10.0
    servo_target_timeout_s: float | None = None
    servo_max_consecutive_overruns: int = 20
    servo_shutdown_timeout_s: float = 2.0

    # The JAKA gripper controller accepts its opening width through extension
    # analog-output channel 3: 0 is fully closed and 1000 is fully open.
    # Input feedback remains optional because this controller does not expose
    # position feedback through the command channel.
    gripper_analog_input_enabled: bool = False
    gripper_analog_output_enabled: bool = True
    gripper_analog_input_iotype: int = IO_CABINET
    gripper_analog_input_index: int = 0
    gripper_analog_output_iotype: int = IO_EXTEND
    gripper_analog_output_index: int = 3
    gripper_analog_input_min: float = 0.0
    gripper_analog_input_max: float = 10.0
    gripper_analog_output_min: float = 0.0
    gripper_analog_output_max: float = 1000.0
    gripper_analog_input_inverted: bool = False
    gripper_analog_output_inverted: bool = False
    gripper_fallback_position: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.ip.strip():
            raise ValueError("JakaRobotConfig.ip must not be empty.")
        if not 0 <= self.tool_id <= 15 or not 0 <= self.user_frame_id <= 15:
            raise ValueError("JAKA tool_id and user_frame_id must be in [0, 15].")
        if not 0 <= self.collision_level <= 5:
            raise ValueError("JakaRobotConfig.collision_level must be in [0, 5].")
        if self.joint_speed <= 0 or self.linear_speed_mm_s <= 0:
            raise ValueError("JAKA planner speeds must be positive.")
        if self.max_eef_step_m <= 0 or self.max_eef_step_rad <= 0:
            raise ValueError("JAKA Cartesian safety limits must be positive.")
        if self.servo_step_num < 1:
            raise ValueError("JakaRobotConfig.servo_step_num must be at least 1.")
        if self.servo_filter_mode not in {"none", "joint_lpf", "joint_nlf", "cartesian_nlf"}:
            raise ValueError("servo_filter_mode must be one of: none, joint_lpf, joint_nlf, cartesian_nlf.")
        if not 1 <= self.servo_queue_warn_depth <= SERVO_QUEUE_MAX:
            raise ValueError(f"servo_queue_warn_depth must be in [1, {SERVO_QUEUE_MAX}].")
        if not 0 < self.servo_joint_max_velocity_rad_s <= math.pi:
            raise ValueError("servo_joint_max_velocity_rad_s must be in (0, pi].")
        for name, value in (
            ("servo_joint_max_acceleration_rad_s2", self.servo_joint_max_acceleration_rad_s2),
            ("servo_eef_max_velocity_m_s", self.servo_eef_max_velocity_m_s),
            ("servo_eef_max_acceleration_m_s2", self.servo_eef_max_acceleration_m_s2),
            ("servo_eef_max_angular_velocity_rad_s", self.servo_eef_max_angular_velocity_rad_s),
            (
                "servo_eef_max_angular_acceleration_rad_s2",
                self.servo_eef_max_angular_acceleration_rad_s2,
            ),
            ("servo_filter_joint_lpf_cutoff_hz", self.servo_filter_joint_lpf_cutoff_hz),
            ("servo_filter_joint_max_jerk_rad_s3", self.servo_filter_joint_max_jerk_rad_s3),
            ("servo_filter_eef_max_jerk_m_s3", self.servo_filter_eef_max_jerk_m_s3),
            (
                "servo_filter_eef_max_angular_jerk_rad_s3",
                self.servo_filter_eef_max_angular_jerk_rad_s3,
            ),
            ("servo_shutdown_timeout_s", self.servo_shutdown_timeout_s),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.servo_target_timeout_s is not None and (
            not np.isfinite(self.servo_target_timeout_s) or self.servo_target_timeout_s <= 0
        ):
            raise ValueError("servo_target_timeout_s must be positive and finite, or None.")
        if self.servo_max_consecutive_overruns < 1:
            raise ValueError("servo_max_consecutive_overruns must be at least 1.")
        if self.max_relative_target is not None and not (
            isinstance(self.max_relative_target, (int, float, dict))
            and not isinstance(self.max_relative_target, bool)
        ):
            raise ValueError("max_relative_target must be a positive number, a joint mapping, or None.")
        if isinstance(self.max_relative_target, (int, float)):
            if not np.isfinite(self.max_relative_target) or self.max_relative_target <= 0:
                raise ValueError("max_relative_target must be positive.")
            self.max_relative_target = float(self.max_relative_target)
        elif isinstance(self.max_relative_target, dict):
            expected_keys = {f"joint_{index}.pos" for index in range(1, 7)}
            if set(self.max_relative_target) != expected_keys:
                raise ValueError(
                    "max_relative_target joint mapping must contain exactly joint_1.pos through joint_6.pos."
                )
            if any(
                not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
                for value in self.max_relative_target.values()
            ):
                raise ValueError("max_relative_target joint mapping values must be positive numbers.")
        self.joint_position_limits = self._validate_position_limits(
            "joint_position_limits",
            self.joint_position_limits,
            {f"joint_{index}.pos" for index in range(1, 7)},
        )
        self.eef_pose_limits = self._validate_position_limits(
            "eef_pose_limits",
            self.eef_pose_limits,
            {"ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw"},
        )
        if not 0.0 <= self.gripper_fallback_position <= 1.0:
            raise ValueError("gripper_fallback_position must be in [0, 1].")

        for name, value in (
            ("gripper_analog_input_iotype", self.gripper_analog_input_iotype),
            ("gripper_analog_output_iotype", self.gripper_analog_output_iotype),
        ):
            if value not in {IO_CABINET, IO_TOOL, IO_EXTEND}:
                raise ValueError(f"{name} must be cabinet (0), tool (1), or extension (2).")
        for name, minimum, maximum in (
            ("gripper_analog_input", self.gripper_analog_input_min, self.gripper_analog_input_max),
            ("gripper_analog_output", self.gripper_analog_output_min, self.gripper_analog_output_max),
        ):
            if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
                raise ValueError(f"{name} range must be finite and increasing.")

    @staticmethod
    def _validate_position_limits(
        name: str,
        limits: dict[str, tuple[float, float]],
        allowed_keys: set[str],
    ) -> dict[str, tuple[float, float]]:
        validated: dict[str, tuple[float, float]] = {}
        for key, bounds in limits.items():
            if key not in allowed_keys:
                raise ValueError(f"{name} contains unsupported field {key!r}.")
            if len(bounds) != 2:
                raise ValueError(f"{name}[{key!r}] must contain [minimum, maximum].")
            minimum, maximum = map(float, bounds)
            if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
                raise ValueError(f"{name}[{key!r}] must be finite and increasing.")
            validated[key] = (minimum, maximum)
        return validated


class JakaRobot(Robot):
    """A LeRobot adapter that drives a JAKA arm through continuous Servo Move.

    ``send_action`` is ready after ``connect`` by default. Each action selects
    `servo_j` when it contains joint fields or `servo_p` when it contains
    Cartesian fields, with target limiting applied inside this adapter. For
    one-off controller-planned moves use ``move_j`` or ``move_l``. Advanced SDK
    calls are intentionally not mirrored: access the connected vendor handle
    through ``rc`` when needed.
    """

    config_class: ClassVar[type] = JakaRobotConfig
    name: ClassVar[str] = "jaka_robot"
    motors: ClassVar[list[str]] = [f"joint_{index}" for index in range(1, 7)]
    _JOINT_KEYS: ClassVar[tuple[str, ...]] = tuple(f"{motor}.pos" for motor in motors)
    _EEF_KEYS: ClassVar[tuple[str, ...]] = (
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.roll",
        "ee.pitch",
        "ee.yaw",
    )

    def __init__(self, config: JakaRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}
        self.rc: Any | None = None
        self._feedback_rc: Any | None = None
        self._sdk_lock = threading.RLock()
        self._feedback_sdk_lock = threading.RLock()
        self._observation_lock = threading.RLock()
        self._last_arm_observation: dict[str, float] | None = None
        self._last_arm_observation_at: float | None = None
        self._servo_state_lock = threading.RLock()
        self._servo_stop = threading.Event()
        self._servo_thread: threading.Thread | None = None
        self._servo_startup_ready: threading.Event | None = None
        self._servo_startup_error: Exception | None = None
        self._servo_active = False
        self._servo_queue_warned = False
        self._last_eef_target: np.ndarray | None = None
        self._servo_representation: Literal["joints", "eef"] | None = None
        self._servo_target: np.ndarray | None = None
        self._servo_commanded_position: np.ndarray | None = None
        self._servo_joint_velocity = np.zeros(6)
        self._servo_linear_velocity = np.zeros(3)
        self._servo_angular_speed = 0.0
        self._servo_target_updated_at: float | None = None
        self._servo_period_samples: deque[float] = deque(maxlen=512)
        self._servo_frames_sent = 0
        self._servo_overruns = 0
        self._servo_consecutive_overruns = 0
        self._servo_queue_depth: int | None = None
        self._servo_last_error: str | None = None
        self._servo_watchdog: str | None = None
        self._last_gripper_position = config.gripper_fallback_position
        self._powered_by_driver = False
        self._enabled_by_driver = False

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {f"{motor}.pos": float for motor in self.motors}
        features["gripper.pos"] = float
        features.update(dict.fromkeys(self._EEF_KEYS, float))
        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{name}_depth"] = (camera.height, camera.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        features: dict[str, Any] = dict.fromkeys(self._JOINT_KEYS, float)
        features["gripper.pos"] = float
        features.update(dict.fromkeys(self._EEF_KEYS, float))
        return features

    @property
    def is_connected(self) -> bool:
        return self.rc is not None and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def servo_active(self) -> bool:
        """Return whether the managed Servo sender is active."""

        with self._servo_state_lock:
            return self._servo_active

    def calibrate(self) -> None:
        """JAKA joint calibration is controller-managed."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Log in, apply configured controller settings, and attach cameras."""

        del calibrate
        rc = create_rc(self.config.ip)
        self._login(rc)
        self.rc = rc
        try:
            self.rc.set_vibsuppress_mode(1)  # enable vibration suppression
            self.rc.vibsuppress_on(12)  # enable vibration suppression

            if self.config.separate_feedback_connection:
                feedback_rc = create_rc(self.config.ip)
                self._login(feedback_rc)
                self._feedback_rc = feedback_rc
            controller_state = self.get_controller_state()
            if self.config.auto_power_on and not controller_state["powered_on"]:
                self._call("power_on", rc.power_on())
                self._powered_by_driver = True
                controller_state["powered_on"] = True
            if self.config.auto_enable and not controller_state["enabled"]:
                self._call("enable_robot", rc.enable_robot())
                self._enabled_by_driver = True

            self._call("set_tool_id", rc.set_tool_id(self.config.tool_id))
            self._call("set_user_frame_id", rc.set_user_frame_id(self.config.user_frame_id))
            self._call("set_collision_level", rc.set_collision_level(self.config.collision_level))
            self.configure()
            for camera in self.cameras.values():
                camera.connect()
            if self.config.auto_enable_servo:
                self.servo_enable(True)
        except BaseException:
            self.disconnect()
            raise
        logger.info("JAKA controller %s connected.", self.config.ip)

    def configure(self) -> None:
        """Use JAKA's T planner for explicit point-to-point movements."""

        if self.rc is not None:
            self._call("set_motion_planner", self.rc.set_motion_planner(PLANNER_S))
            self._configure_servo_filter()

    def _configure_servo_filter(self) -> None:
        """Configure the SDK's Servo Move jitter filter before Servo is enabled."""

        assert self.rc is not None
        mode = self.config.servo_filter_mode
        if mode == "none":
            return

        with self._sdk_lock:
            if mode == "joint_lpf":
                result = self.rc.servo_move_use_joint_LPF(self.config.servo_filter_joint_lpf_cutoff_hz)
                self._call("servo_move_use_joint_LPF", result)
            elif mode == "joint_nlf":
                result = self.rc.servo_move_use_joint_NLF(
                    self.config.servo_joint_max_velocity_rad_s,
                    self.config.servo_joint_max_acceleration_rad_s2,
                    self.config.servo_filter_joint_max_jerk_rad_s3,
                )
                self._call("servo_move_use_joint_NLF", result)
            elif mode == "cartesian_nlf":
                # JAKA's Cartesian NLF uses mm for translation and degrees for
                # orientation; the LeRobot-facing config uses metres and radians.
                result = self.rc.servo_move_use_carte_NLF(
                    self.config.servo_eef_max_velocity_m_s * 1000.0,
                    self.config.servo_eef_max_acceleration_m_s2 * 1000.0,
                    self.config.servo_filter_eef_max_jerk_m_s3 * 1000.0,
                    math.degrees(self.config.servo_eef_max_angular_velocity_rad_s),
                    math.degrees(self.config.servo_eef_max_angular_acceleration_rad_s2),
                    math.degrees(self.config.servo_filter_eef_max_angular_jerk_rad_s3),
                )
                self._call("servo_move_use_carte_NLF", result)

    def disconnect(self) -> None:
        """Safely leave Servo Move, release owned state, and log out."""

        rc = self.rc
        if rc is None:
            return
        feedback_rc = self._feedback_rc
        try:
            if self._servo_active:
                self._set_servo(False, suppress_errors=True)
            if self._enabled_by_driver and self.config.disable_on_disconnect:
                self._call("disable_robot", rc.disable_robot())
            if self._powered_by_driver and self.config.power_off_on_disconnect:
                self._call("power_off", rc.power_off())
            if feedback_rc is not None and feedback_rc is not rc:
                logout_feedback = getattr(feedback_rc, "logout", None) or getattr(
                    feedback_rc, "log_out", None
                )
                if logout_feedback is not None:
                    self._call("feedback_logout", logout_feedback())
            logout = getattr(rc, "logout", None) or getattr(rc, "log_out", None)
            if logout is not None:
                self._call("logout", logout())
        except JakaError as exc:
            logger.warning("JAKA disconnect cleanup failed: %s", exc)
        finally:
            for camera in self.cameras.values():
                try:
                    camera.disconnect()
                except Exception as exc:  # nosec B110 - teardown is best effort
                    logger.warning("JAKA camera cleanup failed: %s", exc)
            self.rc = None
            self._feedback_rc = None
            self._servo_active = False
            with self._observation_lock:
                self._last_arm_observation = None
                self._last_observation_at = None
            self._last_eef_target = None
            self._powered_by_driver = False
            self._enabled_by_driver = False

    def _login(self, rc: Any) -> None:
        login = getattr(rc, "login", None) or getattr(rc, "log_in", None)
        if login is None:
            raise RuntimeError("Installed JAKA binding does not expose login() or log_in().")
        try:
            result = login(use_grpc=int(self.config.use_grpc))
        except TypeError:
            result = login()
        _payload("login", result)

    @staticmethod
    def _call(operation: str, result: Any) -> tuple[Any, ...]:
        return _payload(operation, result)

    def _read_joint_vector(self) -> np.ndarray:
        rc, lock = self._feedback_handle()
        with lock:
            values = self._call("get_actual_joint_position", rc.get_actual_joint_position())
        try:
            return _vector(values, name="actual joint position")
        except ValueError as exc:
            raise JakaError("get_actual_joint_position", -1, values) from exc

    def _read_tcp_vector_mm(self) -> np.ndarray:
        rc, lock = self._feedback_handle()
        with lock:
            values = self._call("get_actual_tcp_position", rc.get_actual_tcp_position())
        try:
            return _vector(values, name="actual TCP position")
        except ValueError as exc:
            raise JakaError("get_actual_tcp_position", -1, values) from exc

    def _feedback_handle(self) -> tuple[Any, threading.RLock]:
        """Return the auxiliary feedback/IO SDK handle and its lock.

        A dedicated handle is optional for compatibility with existing clients,
        but recorder workloads use it to keep feedback RPCs off the Servo handle.
        """

        if self.rc is None:
            raise RuntimeError("Connect the JAKA robot before reading feedback.")
        if self._feedback_rc is not None:
            return self._feedback_rc, self._feedback_sdk_lock
        return self.rc, self._sdk_lock

    @check_if_not_connected
    def get_joint_positions(self) -> dict[str, float]:
        """Return measured joint angles in radians."""

        joints = self._read_joint_vector()
        return {f"{motor}.pos": float(joints[index]) for index, motor in enumerate(self.motors)}

    @check_if_not_connected
    def get_eef_pose(self) -> np.ndarray:
        """Return actual TCP pose as ``[x, y, z, rx, ry, rz]`` in m/rad."""

        pose = self._read_tcp_vector_mm()
        pose[:3] /= 1000.0
        return pose

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read actual joint/TCP state, optional gripper feedback, and cameras."""

        joints = self.get_joint_positions()
        eef = self.get_eef_pose()
        observation: dict[str, Any] = dict(joints)
        observation.update(zip(self._EEF_KEYS, map(float, eef), strict=True))
        observation.update(self.get_gripper_position())
        with self._observation_lock:
            self._last_arm_observation = {
                key: float(value)
                for key, value in observation.items()
                if key in self._JOINT_KEYS or key in self._EEF_KEYS or key == "gripper.pos"
            }
            self._last_observation_at = time.monotonic()
        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                observation[name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                observation[f"{name}_depth"] = camera.read_latest_depth()
        return observation

    def _cached_arm_observation(self, *, max_age_s: float = 0.1) -> dict[str, float] | None:
        with self._observation_lock:
            if self._last_arm_observation is None or self._last_observation_at is None:
                return None
            if time.monotonic() - self._last_observation_at > max_age_s:
                return None
            return dict(self._last_arm_observation)

    @check_if_not_connected
    def send_action(self, action: RobotAction, *, use_servo: bool = True) -> RobotAction:
        """Send one bounded joint, Cartesian, or gripper-only action.

        The arm representation is inferred from the supplied fields. Joint and
        Cartesian fields cannot be mixed in one frame because the controller
        accepts only one arm representation per command. ``use_servo=True``
        atomically updates the target tracked by the internal 8 ms Servo
        scheduler. ``False`` submits one controller-planned joint or linear
        move and requires Servo Move to be inactive.
        """

        representation = self._action_representation(action, name="action")
        if representation is not None and use_servo and not self._servo_active:
            with self._servo_state_lock:
                worker_error = self._servo_last_error
            detail = f" Last worker error: {worker_error}" if worker_error else ""
            raise RuntimeError(
                "Servo Move is disabled; call servo_enable(True) before send_action()." + detail
            )
        if representation is not None and not use_servo and self._servo_active:
            raise RuntimeError("Exit Servo Move before sending a controller-planned action.")
        if "gripper.pos" in action:
            gripper = self.set_gripper_position(action["gripper.pos"])
        else:
            gripper = self.get_gripper_position()["gripper.pos"]
        cached = self._cached_arm_observation()

        if representation == "joints":
            applied = self._send_joint_action(action, use_servo=use_servo, current=cached)
            measured_eef = (
                np.asarray([cached[key] for key in self._EEF_KEYS], dtype=float)
                if cached is not None and all(key in cached for key in self._EEF_KEYS)
                else self.get_eef_pose()
            )
            applied.update(zip(self._EEF_KEYS, map(float, measured_eef), strict=True))
        elif representation == "eef":
            applied = (
                {key: cached[key] for key in self._JOINT_KEYS if key in cached}
                if cached is not None
                else self.get_joint_positions()
            )
            measured_eef = (
                np.asarray([cached[key] for key in self._EEF_KEYS], dtype=float)
                if cached is not None and all(key in cached for key in self._EEF_KEYS)
                else None
            )
            applied.update(self._send_eef_action(action, use_servo=use_servo, measured_eef=measured_eef))
        else:
            applied = (
                {key: cached[key] for key in self._JOINT_KEYS if key in cached}
                if cached is not None and all(key in cached for key in self._JOINT_KEYS)
                else self.get_joint_positions()
            )
            measured_eef = (
                np.asarray([cached[key] for key in self._EEF_KEYS], dtype=float)
                if cached is not None and all(key in cached for key in self._EEF_KEYS)
                else self.get_eef_pose()
            )
            applied.update(zip(self._EEF_KEYS, map(float, measured_eef), strict=True))
        applied["gripper.pos"] = gripper
        return applied

    @check_if_not_connected
    def send_relative_action(self, delta: RobotAction, *, use_servo: bool = True) -> RobotAction:
        """Apply finite deltas through Servo Move or one planned controller move."""

        representation = self._action_representation(delta, name="relative action")
        if representation == "joints":
            target = self.get_joint_positions()
        elif representation == "eef":
            measured_eef = self.get_eef_pose()
            target = dict(zip(self._EEF_KEYS, map(float, measured_eef), strict=True))
            # Cartesian limiting must also use this measured state as its
            # reference, rather than a previous UI or policy command.
            self._last_eef_target = measured_eef if use_servo else None
        else:
            target = {}

        if "gripper.pos" in delta:
            target.update(self.get_gripper_position())

        for key, raw_delta in delta.items():
            try:
                value = float(raw_delta)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"delta[{key!r}] must be numeric") from exc
            if not np.isfinite(value):
                raise ValueError(f"delta[{key!r}] must be finite")
            target[key] += value
        return self.send_action(target, use_servo=use_servo)

    def _action_representation(self, action: RobotAction, *, name: str) -> Literal["joints", "eef"] | None:
        fields = set(action)
        unknown = fields - set(self.action_features)
        if unknown:
            raise ValueError(f"{name} contains unsupported fields: {sorted(unknown)}")

        has_joints = not fields.isdisjoint(self._JOINT_KEYS)
        has_eef = not fields.isdisjoint(self._EEF_KEYS)
        if has_joints and has_eef:
            raise ValueError(f"{name} cannot mix joint and EEF fields in one arm command")
        if has_joints:
            return "joints"
        if has_eef:
            return "eef"
        return None

    def _send_joint_action(
        self,
        action: RobotAction,
        *,
        use_servo: bool,
        current: dict[str, float] | None = None,
    ) -> RobotAction:
        # A following Cartesian action must start from measured feedback, not a
        # target cached before this joint-space command.
        self._last_eef_target = None
        if current is None or not all(key in current for key in self._JOINT_KEYS):
            current = self.get_joint_positions()
        else:
            current = {key: float(current[key]) for key in self._JOINT_KEYS}
        requested = dict(current)
        for motor in self.motors:
            key = f"{motor}.pos"
            if key in action:
                try:
                    value = float(action[key])
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"action[{key!r}] must be numeric") from exc
                if not np.isfinite(value):
                    raise ValueError(f"action[{key!r}] must be finite")
                requested[key] = value
        if self.config.max_relative_target is not None:
            requested = ensure_safe_goal_position(
                {key: (requested[key], current[key]) for key in requested}, self.config.max_relative_target
            )
        requested = self._apply_position_limits(requested, self.config.joint_position_limits)
        target = tuple(requested[f"{motor}.pos"] for motor in self.motors)
        if use_servo:
            self._update_servo_target(
                "joints", np.asarray(target), current=np.fromiter(current.values(), float)
            )
        else:
            self.move_j(target, is_block=True)
        return requested

    def _send_eef_action(
        self,
        action: RobotAction,
        *,
        use_servo: bool,
        measured_eef: np.ndarray | None = None,
    ) -> RobotAction:
        if use_servo and self._last_eef_target is not None:
            reference = self._last_eef_target
        elif measured_eef is not None:
            reference = measured_eef.copy()
        else:
            reference = self.get_eef_pose()
        target = reference.copy()
        for index, key in enumerate(self._EEF_KEYS):
            if key in action:
                try:
                    value = float(action[key])
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"action[{key!r}] must be numeric") from exc
                if not np.isfinite(value):
                    raise ValueError(f"action[{key!r}] must be finite")
                target[index] = value
        delta = target - reference
        translation = float(np.linalg.norm(delta[:3]))
        if translation > self.config.max_eef_step_m:
            target[:3] = reference[:3] + delta[:3] * (self.config.max_eef_step_m / translation)

        # Limit orientation on SO(3), rather than clipping Euler components.
        # Component-wise clipping turns a small +/-pi wrap into a long
        # rotation and is especially visible with noisy XR RPY targets.
        reference_q = _euler_to_quaternion(reference[3:])
        target_q = _euler_to_quaternion(target[3:])
        target_q, angle = _quaternion_distance(reference_q, target_q)
        if angle > self.config.max_eef_step_rad:
            target_q = _quaternion_slerp(
                reference_q,
                target_q,
                self.config.max_eef_step_rad / angle,
            )
            target[3:] = _quaternion_to_euler(target_q, reference_rpy=reference[3:])
        for index, key in enumerate(self._EEF_KEYS):
            if key in self.config.eef_pose_limits:
                target[index] = np.clip(target[index], *self.config.eef_pose_limits[key])
        if use_servo:
            self._update_servo_target("eef", target, current=reference)
            self._last_eef_target = target
        else:
            self.move_l(target, is_block=True)
            self._last_eef_target = None
        return dict(zip(self._EEF_KEYS, map(float, target), strict=True))

    @staticmethod
    def _apply_position_limits(
        positions: dict[str, float], limits: dict[str, tuple[float, float]]
    ) -> dict[str, float]:
        return {
            key: float(np.clip(value, *limits[key])) if key in limits else value
            for key, value in positions.items()
        }

    @check_if_not_connected
    def move_j(self, joints: Any, *, is_block: bool = True, speed: float | None = None) -> None:
        """Execute one controller-planned absolute joint move in radians."""

        target = _vector(joints, name="joints")
        assert self.rc is not None
        with self._sdk_lock:
            self._call(
                "joint_move",
                self.rc.joint_move(
                    tuple(map(float, target)),
                    ABS,
                    bool(is_block),
                    self.config.joint_speed if speed is None else speed,
                ),
            )

    @check_if_not_connected
    def move_l(self, pose: Any, *, is_block: bool = True, speed_mm_s: float | None = None) -> None:
        """Execute one controller-planned absolute TCP linear move in m/rad."""

        target = _vector(pose, name="pose").copy()
        target[:3] *= 1000.0  # convert to mm for JAKA SDK
        assert self.rc is not None
        with self._sdk_lock:
            self._call(
                "linear_move",
                self.rc.linear_move(
                    tuple(map(float, target)),
                    ABS,
                    bool(is_block),
                    self.config.linear_speed_mm_s if speed_mm_s is None else speed_mm_s,
                ),
            )

    @check_if_not_connected
    def set_eef_pose(self, pose: Any) -> None:
        """Compatibility alias for a blocking controller-planned linear move."""

        self.move_l(pose, is_block=False)

    @check_if_not_connected
    def move_eef_relative(self, delta: Any) -> tuple[np.ndarray, np.ndarray]:
        """Move from the measured TCP pose by a relative m/rad offset."""

        current = self.get_eef_pose()
        target = current + _vector(delta, name="delta")
        self.move_l(target, is_block=False)
        return current, target

    @check_if_not_connected
    def power_on(self) -> None:
        """Power the controller on and record that this adapter owns that state."""

        assert self.rc is not None
        with self._sdk_lock:
            self._call("power_on", self.rc.power_on())
        self._powered_by_driver = True

    @check_if_not_connected
    def power_off(self) -> None:
        """Explicitly power the controller off after all motion has stopped."""

        assert self.rc is not None
        with self._sdk_lock:
            self._call("power_off", self.rc.power_off())
        self._powered_by_driver = False

    @check_if_not_connected
    def enable_robot(self) -> None:
        """Enable robot servos and record that this adapter owns that state."""

        assert self.rc is not None
        with self._sdk_lock:
            self._call("enable_robot", self.rc.enable_robot())
        self._enabled_by_driver = True

    @check_if_not_connected
    def disable_robot(self) -> None:
        """Disable robot servos. Servo Move is exited first when needed."""

        if self._servo_active:
            self._set_servo(False)
        assert self.rc is not None
        with self._sdk_lock:
            self._call("disable_robot", self.rc.disable_robot())
        self._enabled_by_driver = False

    @check_if_not_connected
    def get_controller_state(self) -> dict[str, bool]:
        """Return controller-reported power and enable state."""

        rc, lock = self._feedback_handle()
        getter = getattr(rc, "get_robot_status_simple", None)
        if getter is None:
            return {
                "powered_on": self._powered_by_driver,
                "enabled": self._enabled_by_driver,
            }
        with lock:
            values = self._call("get_robot_status_simple", getter())
        if len(values) < 2:
            raise JakaError("get_robot_status_simple", -1, values)
        return {"powered_on": bool(values[-2]), "enabled": bool(values[-1])}

    @check_if_not_connected
    def motion_abort(self) -> None:
        """Stop the controller's active planned motion."""

        assert self.rc is not None
        with self._sdk_lock:
            self._call("motion_abort", self.rc.motion_abort())

    @check_if_not_connected
    def clear_error(self) -> None:
        """Clear a recoverable controller fault after its physical cause is resolved."""

        assert self.rc is not None
        with self._sdk_lock:
            self._call("clear_error", self.rc.clear_error())

    @check_if_not_connected
    def set_drag_mode(self, enabled: bool) -> None:
        """Enable or disable JAKA drag mode."""

        assert self.rc is not None
        with self._sdk_lock:
            self._call("drag_mode_enable", self.rc.drag_mode_enable(bool(enabled)))

    def servo_frame_period_s(self, step_num: int | None = None) -> float:
        """Return the nominal period of one Servo Move frame."""

        step = self.config.servo_step_num if step_num is None else int(step_num)
        if step < 1:
            raise ValueError("step_num must be at least 1.")
        return step * SERVO_CYCLE_S

    @check_if_not_connected
    def servo_enable(
        self, enabled: bool = True, *, representation: Literal["joints", "eef"] = "joints"
    ) -> None:
        """Start or stop continuous 8 ms Servo target tracking.

        Enabling initializes the scheduler from measured joint or Cartesian
        feedback before entering Servo Move. Disabling first stops and joins
        the sender, then asks the controller to leave Servo Move mode.
        Repeated calls are idempotent.
        """

        self._set_servo(bool(enabled), representation=representation)

    def _set_servo(
        self,
        enabled: bool,
        *,
        representation: Literal["joints", "eef"] = "joints",
        suppress_errors: bool = False,
    ) -> None:
        assert self.rc is not None
        if enabled:
            if self._servo_active:
                return
            if representation == "joints":
                initial_position = self._read_joint_vector()
            else:
                initial_position = self.get_eef_pose()
            try:
                self._set_controller_servo(True)
            except JakaError:
                if suppress_errors:
                    logger.warning("Could not enter JAKA Servo Move mode during setup.", exc_info=True)
                    return
                raise
            startup_ready = threading.Event()
            with self._servo_state_lock:
                self._servo_active = True
                self._servo_startup_ready = startup_ready
                self._servo_startup_error = None
                self._servo_queue_warned = False
                self._last_eef_target = None
                self._servo_representation = representation
                self._servo_target = initial_position.copy()
                self._servo_commanded_position = initial_position.copy()
                self._servo_joint_velocity.fill(0.0)
                self._servo_linear_velocity.fill(0.0)
                self._servo_angular_speed = 0.0
                self._servo_target_updated_at = time.monotonic()
                self._servo_period_samples.clear()
                self._servo_frames_sent = 0
                self._servo_overruns = 0
                self._servo_consecutive_overruns = 0
                self._servo_queue_depth = None
                self._servo_last_error = None
                self._servo_watchdog = None
                self._servo_stop.clear()
                self._servo_thread = threading.Thread(
                    target=self._servo_worker,
                    name=f"jaka-servo-{self.config.ip}",
                    daemon=True,
                )
                self._servo_thread.start()
            if not startup_ready.wait(timeout=self.config.servo_shutdown_timeout_s):
                with self._servo_state_lock:
                    startup_error = self._servo_startup_error
                self._set_servo(False, suppress_errors=True)
                if startup_error is not None:
                    raise startup_error
                raise RuntimeError(
                    "JAKA Servo Move did not send an initial frame within "
                    f"{self.config.servo_shutdown_timeout_s:.2f}s."
                )
            with self._servo_state_lock:
                startup_error = self._servo_startup_error
            if startup_error is not None:
                raise startup_error
            return

        thread = self._servo_thread
        if not self._servo_active and (thread is None or not thread.is_alive()):
            return
        self._servo_stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.config.servo_shutdown_timeout_s)
            if thread.is_alive():
                message = (
                    f"Servo sender did not stop within {self.config.servo_shutdown_timeout_s:.2f}s; "
                    "controller mode was not changed concurrently with an in-flight SDK call."
                )
                with self._servo_state_lock:
                    self._servo_last_error = message
                if suppress_errors:
                    logger.error(message)
                    return
                raise RuntimeError(message)
        try:
            self._set_controller_servo(False)
        except JakaError:
            if suppress_errors:
                logger.warning("Could not leave JAKA Servo Move mode during cleanup.", exc_info=True)
            else:
                raise
        finally:
            with self._servo_state_lock:
                self._servo_active = False
                self._servo_thread = None
                self._servo_startup_ready = None
                self._servo_startup_error = None
                self._servo_representation = None
                self._servo_target = None
                self._servo_commanded_position = None
                self._servo_queue_warned = False
                self._last_eef_target = None

    def _set_controller_servo(self, enabled: bool) -> None:
        assert self.rc is not None
        with self._sdk_lock:
            try:
                result = self.rc.servo_move_enable(enabled, True)
            except TypeError:
                result = self.rc.servo_move_enable(enabled)
            self._call("servo_move_enable", result)

    def _update_servo_target(
        self,
        representation: Literal["joints", "eef"],
        target: np.ndarray,
        *,
        current: np.ndarray,
    ) -> None:
        """Atomically replace the desired target consumed by the sender."""

        target = _vector(target, name="Servo target").copy()
        current = _vector(current, name="current Servo position").copy()
        with self._servo_state_lock:
            if not self._servo_active or self._servo_thread is None or not self._servo_thread.is_alive():
                detail = f" Last worker error: {self._servo_last_error}" if self._servo_last_error else ""
                raise RuntimeError(f"Servo Move sender is not running.{detail}")
            if self._servo_representation != representation:
                self._servo_representation = representation
                self._servo_commanded_position = current
                self._servo_joint_velocity.fill(0.0)
                self._servo_linear_velocity.fill(0.0)
                self._servo_angular_speed = 0.0
            self._servo_target = target
            self._servo_target_updated_at = time.monotonic()

    def cancel_eef_motion(self, measured_pose: Any) -> None:
        """Cancel pending Cartesian motion at a measured TCP pose.

        This replaces both sides of the host-side Servo state so the next frame
        is sent directly at the measured pose instead of being clamped relative
        to an older hand target. The controller may still need time to decelerate
        its internal Cartesian NLF trajectory.
        """

        measured = _vector(measured_pose, name="measured TCP pose").copy()
        with self._servo_state_lock:
            if not self._servo_active or self._servo_representation != "eef":
                raise RuntimeError("Cartesian Servo Move must be active to cancel EEF motion.")
            self._servo_target = measured.copy()
            self._servo_commanded_position = measured.copy()
            self._servo_target_updated_at = time.monotonic()
            self._servo_linear_velocity.fill(0.0)
            self._servo_angular_speed = 0.0
            self._last_eef_target = measured.copy()

    def _servo_worker(self) -> None:
        period_s = SERVO_CYCLE_S
        deadline = time.monotonic()
        last_frame_at: float | None = None
        try:
            while not self._servo_stop.is_set():
                if not _wait_for_servo_deadline(deadline, stop=self._servo_stop):
                    return
                frame_at = time.monotonic()
                if last_frame_at is not None:
                    measured_period = frame_at - last_frame_at
                    with self._servo_state_lock:
                        self._servo_period_samples.append(measured_period)
                        if measured_period > period_s * 1.5:
                            self._servo_overruns += 1
                            self._servo_consecutive_overruns += 1
                        else:
                            self._servo_consecutive_overruns = 0
                        if self._servo_consecutive_overruns >= self.config.servo_max_consecutive_overruns:
                            raise RuntimeError(
                                "Servo watchdog detected "
                                f"{self._servo_consecutive_overruns} consecutive timing overruns"
                            )
                last_frame_at = frame_at

                # Compute the next frame while holding state, but never hold
                # it across the blocking SDK call. A foreground action update
                # must be able to replace the target while servo_p is in flight.
                with self._servo_state_lock:
                    self._check_servo_target_timeout(frame_at)
                    representation = self._servo_representation
                    if representation is None or self._servo_target is None:
                        raise RuntimeError("Servo sender has no initialized target")
                    if representation == "joints":
                        frame = self._step_servo_joints(period_s)
                    else:
                        frame = self._step_servo_eef(period_s)

                if representation == "joints":
                    queue_depth = self._send_servo_joint_frame(frame, step_num=1)
                else:
                    queue_depth = self._send_servo_eef_frame(frame, step_num=1)

                with self._servo_state_lock:
                    self._servo_frames_sent += 1
                    self._servo_queue_depth = queue_depth
                    if self._servo_startup_ready is not None:
                        self._servo_startup_ready.set()

                deadline += period_s
                if frame_at >= deadline:
                    missed = int((frame_at - deadline) // period_s) + 1
                    deadline += missed * period_s
        except Exception as exc:
            self._fail_servo_worker(exc)

    def _check_servo_target_timeout(self, now: float) -> None:
        timeout_s = self.config.servo_target_timeout_s
        if timeout_s is None or self._servo_target_updated_at is None:
            return
        age_s = now - self._servo_target_updated_at
        if age_s > timeout_s:
            self._servo_watchdog = f"target timeout after {age_s:.3f}s"
            raise RuntimeError(f"Servo watchdog target timeout after {age_s:.3f}s")

    def _step_servo_joints(self, period_s: float) -> np.ndarray:
        assert self._servo_commanded_position is not None and self._servo_target is not None
        error = self._servo_target - self._servo_commanded_position
        acceleration = self.config.servo_joint_max_acceleration_rad_s2
        max_velocity = self.config.servo_joint_max_velocity_rad_s
        goal_velocity = np.sign(error) * np.minimum(max_velocity, np.sqrt(2.0 * acceleration * np.abs(error)))
        self._servo_joint_velocity += np.clip(
            goal_velocity - self._servo_joint_velocity,
            -acceleration * period_s,
            acceleration * period_s,
        )
        step = self._servo_joint_velocity * period_s
        reached = np.abs(step) >= np.abs(error)
        step[reached] = error[reached]
        self._servo_joint_velocity[reached] = 0.0
        self._servo_commanded_position = self._servo_commanded_position + step
        return self._servo_commanded_position.copy()

    def _step_servo_eef(self, period_s: float) -> np.ndarray:
        assert self._servo_commanded_position is not None and self._servo_target is not None

        if self.config.servo_filter_mode == "cartesian_nlf":
            # The controller-side NLF already applies Cartesian velocity,
            # acceleration, and jerk limits. Interpolating the same target here
            # applies the motion profile twice and creates substantial teleop lag.
            self._servo_commanded_position = self._servo_target.copy()
            self._servo_linear_velocity.fill(0.0)
            self._servo_angular_speed = 0.0
            return self._servo_commanded_position.copy()

        commanded = self._servo_commanded_position.copy()

        translation_error = self._servo_target[:3] - commanded[:3]
        distance = float(np.linalg.norm(translation_error))
        if distance > 1e-12:
            max_velocity = self.config.servo_eef_max_velocity_m_s
            acceleration = self.config.servo_eef_max_acceleration_m_s2
            speed = min(max_velocity, math.sqrt(2.0 * acceleration * distance))
            goal_velocity = translation_error / distance * speed
            self._servo_linear_velocity = _approach_vector(
                self._servo_linear_velocity, goal_velocity, acceleration * period_s
            )
            translation_step = self._servo_linear_velocity * period_s
            if float(np.linalg.norm(translation_step)) >= distance:
                commanded[:3] = self._servo_target[:3]
                self._servo_linear_velocity.fill(0.0)
            else:
                commanded[:3] += translation_step
        else:
            self._servo_linear_velocity.fill(0.0)

        start_q = _euler_to_quaternion(commanded[3:])
        target_q = _euler_to_quaternion(self._servo_target[3:])
        target_q, angle = _quaternion_distance(start_q, target_q)
        if angle > 1e-9:
            acceleration = self.config.servo_eef_max_angular_acceleration_rad_s2
            goal_speed = min(
                self.config.servo_eef_max_angular_velocity_rad_s,
                math.sqrt(2.0 * acceleration * angle),
            )
            self._servo_angular_speed = min(goal_speed, self._servo_angular_speed + acceleration * period_s)
            angular_step = self._servo_angular_speed * period_s
            if angular_step >= angle:
                commanded[3:] = self._servo_target[3:]
                self._servo_angular_speed = 0.0
            else:
                commanded[3:] = _quaternion_to_euler(
                    _quaternion_slerp(start_q, target_q, angular_step / angle),
                    reference_rpy=commanded[3:],
                )
        else:
            self._servo_angular_speed = 0.0

        self._servo_commanded_position = commanded
        return commanded.copy()

    def _fail_servo_worker(self, exc: Exception) -> None:
        message = str(exc)
        logger.error("JAKA Servo sender stopped: %s", message, exc_info=True)
        with self._servo_state_lock:
            self._servo_last_error = message
            if self._servo_startup_ready is not None and not self._servo_startup_ready.is_set():
                self._servo_startup_error = exc
                self._servo_startup_ready.set()
            if self._servo_watchdog is None and "watchdog" in message.lower():
                self._servo_watchdog = message
            self._servo_active = False
            self._servo_stop.set()
        try:
            self._set_controller_servo(False)
        except Exception:
            logger.error("Could not leave Servo Move after sender failure.", exc_info=True)

    def get_servo_status(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of sender timing and safety state."""

        with self._servo_state_lock:
            samples = np.asarray(self._servo_period_samples, dtype=float)
            target = tuple(map(float, self._servo_target)) if self._servo_target is not None else None
            commanded_position = (
                tuple(map(float, self._servo_commanded_position))
                if self._servo_commanded_position is not None
                else None
            )
            mean_period = float(samples.mean()) if samples.size else 0.0
            now = time.monotonic()
            target_age = (
                max(0.0, now - self._servo_target_updated_at)
                if self._servo_target_updated_at is not None
                else None
            )
            return {
                "active": self._servo_active,
                "worker_alive": self._servo_thread is not None and self._servo_thread.is_alive(),
                "representation": self._servo_representation,
                "filter_mode": self.config.servo_filter_mode,
                "target": target,
                "commanded_position": commanded_position,
                "target_age_s": target_age,
                "send_rate_hz": 1.0 / mean_period if mean_period > 0 else 0.0,
                "period_p95_ms": float(np.percentile(samples, 95) * 1000.0) if samples.size else 0.0,
                "period_max_ms": float(samples.max() * 1000.0) if samples.size else 0.0,
                "frames_sent": self._servo_frames_sent,
                "overruns": self._servo_overruns,
                "consecutive_overruns": self._servo_consecutive_overruns,
                "queue_depth": self._servo_queue_depth,
                "watchdog": self._servo_watchdog,
                "last_error": self._servo_last_error,
            }

    @contextmanager
    def servo_stream(self, step_num: int | None = None) -> Generator[float]:
        """Temporarily enter Servo Move mode and yield its expected period."""

        if not self.is_connected:
            raise RuntimeError("Connect the JAKA robot before starting a Servo Move stream.")
        was_active = self._servo_active
        self.servo_enable(True)
        try:
            yield self.servo_frame_period_s(step_num)
        finally:
            if not was_active and self.is_connected:
                self.servo_enable(False)

    @check_if_not_connected
    def is_in_servo(self) -> bool:
        """Return the controller-reported Servo Move state."""

        rc, lock = self._feedback_handle()
        with lock:
            values = self._call("is_in_servomove", rc.is_in_servomove())
        return bool(values[0]) if values else self._servo_active

    @check_if_not_connected
    def servo_joint_frame(
        self, joints: Any, *, move_mode: int = ABS, step_num: int | None = None
    ) -> int | None:
        """Submit one advanced raw joint frame outside target tracking.

        Prefer ``send_action(..., use_servo=True)``. A raw frame is serialized
        with the managed sender but will be superseded on its next 8 ms cycle.
        """

        if not self._servo_active:
            raise RuntimeError("Servo Move is disabled.")
        if move_mode not in {ABS, INCR}:
            raise ValueError("Servo Move supports ABS or INCR targets only.")
        target = _vector(joints, name="joints")
        return self._send_servo_joint_frame(target, move_mode=move_mode, step_num=step_num)

    @check_if_not_connected
    def servo_eef_frame(self, pose: Any, *, move_mode: int = ABS, step_num: int | None = None) -> int | None:
        """Submit one advanced raw Cartesian frame outside target tracking."""

        if not self._servo_active:
            raise RuntimeError("Servo Move is disabled.")
        if move_mode not in {ABS, INCR}:
            raise ValueError("Servo Move supports ABS or INCR targets only.")
        target = _vector(pose, name="pose")
        return self._send_servo_eef_frame(target, move_mode=move_mode, step_num=step_num)

    def _send_servo_joint_frame(
        self, joints: np.ndarray, *, move_mode: int = ABS, step_num: int | None = None
    ) -> int | None:
        assert self.rc is not None
        with self._sdk_lock:
            return self._queue_depth(
                self._call(
                    "servo_j",
                    self.rc.servo_j(tuple(map(float, joints)), move_mode, self._servo_step_num(step_num)),
                )
            )

    def _send_servo_eef_frame(
        self, pose: np.ndarray, *, move_mode: int = ABS, step_num: int | None = None
    ) -> int | None:
        target = pose.copy()
        target[:3] *= 1000.0
        assert self.rc is not None
        with self._sdk_lock:
            return self._queue_depth(
                self._call(
                    "servo_p",
                    self.rc.servo_p(tuple(map(float, target)), move_mode, self._servo_step_num(step_num)),
                )
            )

    def _servo_step_num(self, step_num: int | None) -> int:
        step = self.config.servo_step_num if step_num is None else int(step_num)
        if step < 1:
            raise ValueError("step_num must be at least 1.")
        return step

    def _queue_depth(self, values: tuple[Any, ...]) -> int | None:
        if not values:
            return None
        try:
            depth = int(values[0])
        except (TypeError, ValueError) as exc:
            raise JakaError("servo_move", -1, values) from exc
        if not 0 <= depth <= SERVO_QUEUE_MAX:
            raise JakaError("servo_move", -1, values)
        if depth >= self.config.servo_queue_warn_depth and not self._servo_queue_warned:
            logger.warning("JAKA Servo Move queue depth is %d/%d.", depth, SERVO_QUEUE_MAX)
            self._servo_queue_warned = True
        return depth

    @check_if_not_connected
    def get_gripper_position(self) -> dict[str, float]:
        """Return normalized analog gripper feedback or the configured fallback."""

        if not self.config.gripper_analog_input_enabled:
            return {"gripper.pos": self._last_gripper_position}
        rc, lock = self._feedback_handle()
        with lock:
            values = self._call(
                "get_analog_input",
                rc.get_analog_input(
                    self.config.gripper_analog_input_iotype, self.config.gripper_analog_input_index
                ),
            )
        raw = float(values[0])
        position = (raw - self.config.gripper_analog_input_min) / (
            self.config.gripper_analog_input_max - self.config.gripper_analog_input_min
        )
        if self.config.gripper_analog_input_inverted:
            position = 1.0 - position
        return {"gripper.pos": float(np.clip(position, 0.0, 1.0))}

    @check_if_not_connected
    def set_gripper_position(self, position: Any) -> float:
        """Set normalized gripper opening through JAKA extension analog output 3."""

        try:
            normalized = float(position)
        except (TypeError, ValueError) as exc:
            raise TypeError("gripper position must be numeric") from exc
        if not np.isfinite(normalized):
            raise ValueError("gripper position must be finite")
        normalized = float(np.clip(normalized, 0.0, 1.0))
        applied_position = normalized
        if not self.config.gripper_analog_output_enabled:
            self._last_gripper_position = applied_position
            return normalized
        if self.config.gripper_analog_output_inverted:
            normalized = 1.0 - normalized
        raw = self.config.gripper_analog_output_min + normalized * (
            self.config.gripper_analog_output_max - self.config.gripper_analog_output_min
        )
        rc, lock = self._feedback_handle()
        with lock:
            self._call(
                "set_analog_output",
                rc.set_analog_output(
                    iotype=self.config.gripper_analog_output_iotype,
                    index=self.config.gripper_analog_output_index,
                    value=raw,
                ),
            )
        self._last_gripper_position = applied_position
        return applied_position

    @property
    def controller_ip(self) -> str:
        return self.config.ip

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return f"JakaRobot(ip={self.config.ip!r}, {state})"


__all__ = [
    "ABS",
    "INCR",
    "CONT",
    "COORD_BASE",
    "COORD_JOINT",
    "COORD_TOOL",
    "IO_CABINET",
    "IO_TOOL",
    "IO_EXTEND",
    "PLANNER_DISABLED",
    "PLANNER_T",
    "PLANNER_S",
    "SERVO_CYCLE_S",
    "DEFAULT_SERVO_STEP_NUM",
    "SERVO_QUEUE_MAX",
    "JakaError",
    "JakaRobotConfig",
    "JakaRobot",
]
