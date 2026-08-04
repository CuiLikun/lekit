"""LeRobot-compatible driver for a 6-DOF JAKA robotic arm.

This module wraps the vendored :mod:`jkrc` Python SDK (``libjakaAPI.so`` and
``jkrc.so`` shipped in this directory). On Linux these shared libraries are
loaded via the ``LD_LIBRARY_PATH`` set up at import time below — keep them
next to this file when deploying.

Example:

    >>> from hardwares.jaka_robot import JakaRobot, JakaRobotConfig
    >>> cfg = JakaRobotConfig(ip="192.168.2.64")
    >>> robot = JakaRobot(cfg)
    >>> with robot:
    ...     obs = robot.get_observation()
    ...     robot.send_action({"joint_1.pos": 0.0, "joint_2.pos": 0.5, ...})

See https://www.jaka.com/docs/guide/1.7.2/SDK/Python.html for the upstream
SDK reference (1.7.2 controller family).
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

# Add the directory containing the shared library to the LD_LIBRARY_PATH
# environment variable so ``libjakaAPI.so`` and ``jkrc.so`` can be located by
# the dynamic linker at module-import time. The dynamic linker only consults
# ``LD_LIBRARY_PATH`` at process start, so this only helps if the missing
# libraries are siblings of the .so on ``sys.path``; in most deployments the
# launcher script already sets the variable and this is a no-op.
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["LD_LIBRARY_PATH"] = f"{_script_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Updating LD_LIBRARY_PATH after process startup does not affect the active
# dynamic linker. Load the vendored API with global symbols before jkrc.so.
_jaka_api_path = os.path.join(_script_dir, "libjakaAPI.so")
try:
    ctypes.CDLL(_jaka_api_path, mode=ctypes.RTLD_GLOBAL)
except OSError as e:
    raise ImportError(f"Unable to load the vendored JAKA SDK library at {_jaka_api_path}: {e}") from e

jkrc = importlib.import_module("jkrc")

logger = logging.getLogger(__name__)


# ── SDK Constants ────────────────────────────────────────────────────────────
# The JAKA Python SDK does not export any symbolic constants — every example
# in the vendor docs literalises them. We re-export them here so callers can
# import them by name.

# Move modes used by joint_move / linear_move / circular_move / servo_*.
ABS, INCR, CONT = 0, 1, 2

# Coordinate frames accepted by ``jog`` and friends.
COORD_BASE, COORD_JOINT, COORD_TOOL = 0, 1, 2

# Cartesian jog axes (matches ``aj_num`` 0..5 in cartesian coord types).
CART_X, CART_Y, CART_Z, CART_RX, CART_RY, CART_RZ = 0, 1, 2, 3, 4, 5

# IO group selectors (``iotype`` parameter).
IO_CABINET, IO_TOOL, IO_EXTEND = 0, 1, 2

# Program states returned by ``get_program_state``.
PROG_STOPPED, PROG_RUNNING, PROG_PAUSED = 0, 1, 2

# Motion planner types for ``set_motion_planner``.
PLANNER_DISABLED, PLANNER_T, PLANNER_S = -1, 0, 1

# Servo Move timing and queue limits documented by JAKA SDK 1.7.2.
SERVO_CYCLE_S = 0.008
DEFAULT_SERVO_STEP_NUM = 4
SERVO_QUEUE_MAX = 100

# Selective subset of controller error codes — surfaces a helpful message
# instead of an opaque ``ret[0]`` value. See the JAKA SDK Python docs for the
# complete table.
JAKA_ERROR_MESSAGES: dict[int, str] = {
    -1: "operation failed",
    -2: "invalid argument",
    -3: "communication failure or not connected",
    -4: "kinematic / inverse-kinematics failure",
    -5: "obstacle / collision detected",
    -6: "robot not powered on",
    -7: "robot not enabled",
    -8: "servo mode not enabled",
}


# ── Exception ────────────────────────────────────────────────────────────────


class JakaError(RuntimeError):
    """Raised when a JAKA SDK call returns a non-zero error code.

    Attributes:
        code: The integer error code returned by the SDK (i.e. ``ret[0]``).
        payload: The remainder of the SDK return tuple, when available.
    """

    def __init__(self, code: int, message: str = "", payload: tuple | None = None):
        self.code = int(code)
        self.payload = payload
        msg = message or JAKA_ERROR_MESSAGES.get(self.code, "unknown error")
        super().__init__(f"JAKA SDK error [{self.code}]: {msg}")


# ── Config ───────────────────────────────────────────────────────────────────


@RobotConfig.register_subclass("jaka_robot")
@dataclass
class JakaRobotConfig(RobotConfig):
    """Configuration for :class:`JakaRobot`.

    All fields are keyword-only (inherited from ``RobotConfig``). When loaded
    from YAML, ``type: jaka_robot`` selects this configuration.
    """

    # Controller IP address.
    ip: str = "192.168.1.31"

    # Cameras recorded alongside JAKA state and actions.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Joint mode uses joint_move; ee_pose streams absolute servo_p targets.
    control_mode: str = "joints"

    # Optional analog gripper integration. JAKA's SDK exposes analog input
    # feedback and analog output commands, but the electrical range and channel
    # depend on the installed gripper/controller wiring. Both directions are
    # disabled by default so connecting the arm never changes an unknown AO.
    gripper_analog_input_enabled: bool = False
    gripper_analog_output_enabled: bool = False
    gripper_analog_input_iotype: int = IO_CABINET
    gripper_analog_input_index: int = 0
    gripper_analog_output_iotype: int = IO_CABINET
    gripper_analog_output_index: int = 0
    gripper_analog_input_min: float = 0.0
    gripper_analog_input_max: float = 10.0
    gripper_analog_output_min: float = 0.0
    gripper_analog_output_max: float = 10.0
    gripper_analog_input_inverted: bool = False
    gripper_analog_output_inverted: bool = False
    gripper_fallback_position: float = 0.0

    # Default joint-space motion (rad/s, rad/s²). Used when ``move_j`` callers
    # do not override ``speed`` / ``acc`` explicitly.
    default_speed_joint: float = 0.5
    default_acc_joint: float = 3.5

    # Default Cartesian motion (mm/s, mm/s²).
    default_speed_linear: float = 100.0
    default_acc_linear: float = 500.0

    # Safety clamp applied to every ``send_action``: any joint whose target
    # differs from its current position by more than this magnitude is capped.
    # ``None`` disables clamping.
    max_relative_target: float | dict[str, float] | None = 0.05

    # Per-frame Cartesian Servo Move limits.
    max_eef_step_m: float = 0.01
    max_eef_step_rad: float = 0.08

    # Whether ``connect()`` should also power-on and enable the servos so the
    # first ``send_action`` is immediately ready.
    auto_power_on: bool = True
    auto_enable: bool = True

    # Use the gRPC channel for ``login`` when supported by the controller.
    use_grpc: bool = False

    # Default tool / user-frame IDs applied at connect time.
    tool_id: int = 0
    user_frame_id: int = 0

    # Collision protection level (0=disabled, 1..5 ⇒ 25/50/75/100/125 N).
    collision_level: int = 3

    # ``set_block_wait_timeout`` value used for blocking ``*_move`` calls.
    block_wait_timeout_s: float = 10.0

    # Servo Move frame period is ``servo_step_num * 8 ms``. The default four
    # cycles correspond to a nominal 31.25 Hz controller command period.
    servo_step_num: int = DEFAULT_SERVO_STEP_NUM

    # Warn once when the SDK-reported Servo Move queue reaches this depth.
    servo_queue_warn_depth: int = 80

    def __post_init__(self):
        # Parent ensures cameras (if any) declare width/height/fps. JAKA ships
        # with no cameras, so the loop is a no-op.
        super().__post_init__()
        if not self.ip:
            raise ValueError("JakaRobotConfig.ip must not be empty.")
        if self.control_mode not in ("joints", "ee_pose"):
            raise ValueError("JakaRobotConfig.control_mode must be 'joints' or 'ee_pose'.")
        for field_name, iotype in (
            ("gripper_analog_input_iotype", self.gripper_analog_input_iotype),
            ("gripper_analog_output_iotype", self.gripper_analog_output_iotype),
        ):
            if iotype not in (IO_CABINET, IO_TOOL, IO_EXTEND):
                raise ValueError(
                    f"JakaRobotConfig.{field_name} must be 0 (cabinet), 1 (tool), or 2 (extend)."
                )
        if self.gripper_analog_input_index < 0:
            raise ValueError("JakaRobotConfig.gripper_analog_input_index must be non-negative.")
        if self.gripper_analog_output_index < 0:
            raise ValueError("JakaRobotConfig.gripper_analog_output_index must be non-negative.")
        analog_limits = (
            self.gripper_analog_input_min,
            self.gripper_analog_input_max,
            self.gripper_analog_output_min,
            self.gripper_analog_output_max,
        )
        if not all(np.isfinite(limit) for limit in analog_limits):
            raise ValueError("JakaRobotConfig gripper analog limits must all be finite.")
        if self.gripper_analog_input_max <= self.gripper_analog_input_min:
            raise ValueError(
                "JakaRobotConfig.gripper_analog_input_max must be greater than gripper_analog_input_min."
            )
        if self.gripper_analog_output_max <= self.gripper_analog_output_min:
            raise ValueError(
                "JakaRobotConfig.gripper_analog_output_max must be greater than gripper_analog_output_min."
            )
        if not 0.0 <= self.gripper_fallback_position <= 1.0:
            raise ValueError("JakaRobotConfig.gripper_fallback_position must be in [0, 1].")
        if self.default_speed_joint <= 0:
            raise ValueError("JakaRobotConfig.default_speed_joint must be positive.")
        if self.default_acc_joint <= 0:
            raise ValueError("JakaRobotConfig.default_acc_joint must be positive.")
        if self.default_speed_linear <= 0:
            raise ValueError("JakaRobotConfig.default_speed_linear must be positive.")
        if self.max_relative_target is not None and not (
            (isinstance(self.max_relative_target, (int, float)) and self.max_relative_target > 0)
            or isinstance(self.max_relative_target, dict)
        ):
            raise ValueError(
                "JakaRobotConfig.max_relative_target must be a positive scalar or a "
                "dict of motor_name -> positive scalar, or None to disable."
            )
        if self.max_eef_step_m <= 0:
            raise ValueError("JakaRobotConfig.max_eef_step_m must be positive.")
        if self.max_eef_step_rad <= 0:
            raise ValueError("JakaRobotConfig.max_eef_step_rad must be positive.")
        if not 0 <= self.collision_level <= 5:
            raise ValueError("JakaRobotConfig.collision_level must be in [0, 5].")
        if self.tool_id < 0 or self.tool_id > 15:
            raise ValueError("JakaRobotConfig.tool_id must be in [0, 15].")
        if self.user_frame_id < 0 or self.user_frame_id > 15:
            raise ValueError("JakaRobotConfig.user_frame_id must be in [0, 15].")
        if self.block_wait_timeout_s < 0.5:
            raise ValueError("JakaRobotConfig.block_wait_timeout_s must be >= 0.5.")
        if not 1 <= self.servo_step_num <= DEFAULT_SERVO_STEP_NUM:
            raise ValueError(
                f"JakaRobotConfig.servo_step_num must be in [1, {DEFAULT_SERVO_STEP_NUM}] "
                "to maintain a nominal rate of at least 30 Hz."
            )
        if not 1 <= self.servo_queue_warn_depth <= SERVO_QUEUE_MAX:
            raise ValueError(f"JakaRobotConfig.servo_queue_warn_depth must be in [1, {SERVO_QUEUE_MAX}].")


# ── Robot implementation ─────────────────────────────────────────────────────


class JakaRobot(Robot):
    """LeRobot-compatible JAKA 6-DOF robotic arm driver.

    The class follows the standard :class:`lerobot.robots.robot.Robot`
    contract. ``get_observation`` always returns six measured joint positions,
    normalized gripper feedback, and the measured TCP pose. ``send_action``
    accepts that same fixed arm-state schema regardless of control mode: joint
    mode executes the joint fields, while ee_pose mode streams the Cartesian
    fields through servo_p. The inactive representation is recorded from the
    controller's measured state.

    For low-level access (drag mode, drag-mode IO, sinusoidal jog, servo
    streams, …) the underlying :class:`jkrc.RC` methods are exposed as
    thin wrappers — see the ``move_*``, ``servo_*``, ``set_drag_mode`` and
    IO helpers further down.
    """

    config_class: ClassVar[type] = JakaRobotConfig
    name: ClassVar[str] = "jaka_robot"

    # JAKA arms are 6-DOF. Joints are exposed under stable names so the
    # LeRobot observation/action schemas line up regardless of firmware.
    N_JOINTS: ClassVar[int] = 6
    motors: ClassVar[list[str]] = [f"joint_{i + 1}" for i in range(N_JOINTS)]
    TCP_AXES: ClassVar[tuple[str, ...]] = ("x", "y", "z", "rx", "ry", "rz")
    EEF_ACTION_KEYS: ClassVar[tuple[str, ...]] = (
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.roll",
        "ee.pitch",
        "ee.yaw",
    )

    # ── Construction ──

    def __init__(self, config: JakaRobotConfig):
        super().__init__(config)
        self.config: JakaRobotConfig = config
        self.cameras = make_cameras_from_configs(config.cameras)
        # Underlying SDK controller handle; populated in ``connect``.
        self.rc: Any | None = None
        self._servo_active = False
        self._servo_queue_warned = False
        self._last_eef_command: np.ndarray | None = None

    # ── Feature schemas ──

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """Measured joints, gripper, end-effector pose, and camera frames."""
        features: dict[str, Any] = {f"{m}.pos": float for m in self.motors}
        features["gripper.pos"] = float
        for key in self.EEF_ACTION_KEYS:
            features[key] = float
        features.update({f"{m}.actual_pos": float for m in self.motors})
        for axis in self.TCP_AXES:
            features[f"tcp.{axis}"] = float
            features[f"tcp.actual_{axis}"] = float
        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[camera_name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{camera_name}_depth"] = (camera.height, camera.width, 1)
        features.update(
            {
                "status.powered_on": float,
                "status.enabled": float,
                "status.estop": float,
                "status.collision": float,
                "status.on_limit": float,
                "status.in_pos": float,
                "status.drag_mode": float,
                "status.rapidrate": float,
            }
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """Fixed joint, gripper, and Cartesian action schema."""
        features: dict[str, Any] = {f"{m}.pos": float for m in self.motors}
        features["gripper.pos"] = float
        features.update(dict.fromkeys(self.EEF_ACTION_KEYS, float))
        return features

    # ── Connection state ──

    @property
    def is_connected(self) -> bool:
        """True once the controller and every configured camera are connected."""
        return self.rc is not None and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        # The controller handles joint calibration internally; the host has
        # no per-motor calibration step.
        return True

    def calibrate(self) -> None:
        """No-op: JAKA controllers self-calibrate; nothing to do here."""
        return

    # ── Lifecycle ──

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the controller, log in, optionally power-on and enable.

        Args:
            calibrate: Accepted for interface parity — ignored. JAKA has no
                host-side calibration step to run.
        """
        del calibrate  # Unused — see docstring.

        rc = jkrc.RC(self.config.ip)
        # The vendor docs and the upstream ``login``/``logout`` names disagree
        # with the .pyi stubs (``log_in``/``log_out``). Probe at runtime and
        # fall back so we work with either binding.
        login = getattr(rc, "login", None) or rc.log_in
        use_grpc = 1 if self.config.use_grpc else 0
        try:
            self._check(login(use_grpc=use_grpc))
        except TypeError:
            # The legacy TCP-only binding takes no ``use_grpc`` argument.
            self._check(login())
        logger.info("JakaRobot[%s] logged in (use_grpc=%s)", self.config.ip, self.config.use_grpc)
        self.rc = rc

        if self.config.auto_power_on:
            self.power_on()
        if self.config.auto_enable:
            self.enable_robot()

        # Default IDs / safety settings.
        self._check(self.rc.set_tool_id(self.config.tool_id))
        self._check(self.rc.set_user_frame_id(self.config.user_frame_id))
        self._check(self.rc.set_collision_level(self.config.collision_level))

        try:
            for camera in self.cameras.values():
                camera.connect()
            self.configure()
        except Exception:
            logger.exception("JakaRobot: camera initialization failed; cleaning up connection.")
            self.disconnect()
            raise
        logger.info("JakaRobot[%s] connected and ready.", self.config.ip)

    def configure(self) -> None:
        """Apply runtime configuration. Called once at the end of ``connect``.

        Override-friendly — subclasses may replace this to install custom
        planners, drag-mode friction coefficients, payload settings, etc.
        """
        if not self.is_connected:
            return
        # Default to the T-planner (speed priority). Callers can flip this at
        # runtime via ``set_motion_planner``.
        self._check(self.rc.set_motion_planner(PLANNER_T))

    def disconnect(self) -> None:
        """Best-effort teardown: disable, log out, drop the handle."""
        rc = self.rc
        try:
            if rc is not None and self._servo_active:
                try:
                    self._check(rc.servo_move_enable(False, True))
                except JakaError as e:
                    logger.warning("JakaRobot: could not exit servo mode during disconnect: %s", e)
                finally:
                    self._servo_active = False
                    self._last_eef_command = None
            try:
                # ``payload = (ctrl_errcode, errmsg, powered_on, enabled)``
                if rc is not None:
                    payload = self._check(rc.get_robot_status_simple(), return_payload=True)
                    if bool(payload[3]):
                        self._check(rc.disable_robot())
            except JakaError:
                pass
            if rc is not None:
                logout = getattr(rc, "logout", None) or rc.log_out
                self._check(logout())
        except JakaError as e:
            logger.warning("JakaRobot: error during disconnect: %s", e)
        finally:
            for camera in self.cameras.values():
                if camera.is_connected or getattr(camera, "thread", None) is not None:
                    try:
                        camera.disconnect()
                    except Exception as e:  # nosec B110
                        logger.warning("JakaRobot: could not disconnect camera: %s", e)
            self.rc = None
            self._last_eef_command = None
            logger.info("JakaRobot[%s] disconnected.", self.config.ip)

    # ── Return-tuple validation ──

    def _check(self, ret, return_payload: bool = False):
        """Validate the error code at ``ret[0]``; optionally yield the payload.

        All JAKA SDK calls return a tuple whose first element is an integer
        error code (0 = success). On any non-zero value this raises
        :class:`JakaError`. With ``return_payload=True`` it instead returns
        the data portion of the return value as a flat tuple.

        The SDK uses two success shapes:

            * nested: ``(0, payload_tuple)`` (e.g. ``get_joint_position``)
            * flat:   ``(0, val1, val2, ...)`` (e.g. ``get_rapidrate``)

        For the nested case the inner tuple is unwrapped once, so callers
        always get a flat sequence they can index directly.
        """
        if ret is None:
            raise JakaError(-1, "SDK call returned None")
        # ``jkrc`` returns plain tuples; be defensive about types anyway.
        try:
            code = int(ret[0])
        except (TypeError, IndexError, ValueError) as e:
            raise JakaError(-1, f"unparseable SDK return: {ret!r}") from e
        tail = tuple(ret[1:]) if len(ret) > 1 else ()
        # Unwrap a single nested tuple: ``(0, (a, b, c))`` → ``(a, b, c)``.
        if len(tail) == 1 and isinstance(tail[0], (tuple, list)):
            tail = tuple(tail[0])
        if code != 0:
            raise JakaError(code, payload=tail)
        if return_payload:
            return tail
        return None

    # ── Observation / Action ──

    @check_if_not_connected
    def get_eef_pose(self) -> np.ndarray:
        """Return the actual end-effector pose as ``[x, y, z, rx, ry, rz]``.

        Translation is expressed in metres and XYZ Euler angles in radians.
        """
        payload = self._check(self.rc.get_actual_tcp_position(), return_payload=True)
        if len(payload) != len(self.TCP_AXES):
            raise JakaError(
                -1,
                f"expected {len(self.TCP_AXES)} end-effector pose values, got {len(payload)}",
                payload=payload,
            )
        try:
            pose = np.asarray(payload, dtype=float).copy()
        except (TypeError, ValueError) as e:
            raise JakaError(-1, "end-effector pose contains non-numeric values", payload=payload) from e
        pose[:3] /= 1000.0
        return pose

    @check_if_not_connected
    def set_eef_pose(self, target: list[float] | tuple[float, ...] | np.ndarray) -> None:
        """Move the end effector to ``[x, y, z, rx, ry, rz]``.

        Translation is specified in metres and XYZ Euler angles in radians.
        The target is executed as a blocking absolute Cartesian linear move.
        """
        try:
            pose = np.asarray(target, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError("target must contain six numeric pose values") from e
        if pose.shape != (len(self.TCP_AXES),):
            raise ValueError(f"target must have shape ({len(self.TCP_AXES)},), got {pose.shape}")
        if not np.all(np.isfinite(pose)):
            raise ValueError("target pose values must all be finite")

        tcp_target = pose.copy()
        tcp_target[:3] *= 1000.0
        self.move_l(tcp_target, move_mode=ABS, is_block=True)

    @check_if_not_connected
    def move_eef_relative(
        self, delta: list[float] | tuple[float, ...] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move relative to the latest actual end-effector pose.

        ``delta`` uses metres/radians. The method samples the actual TCP pose,
        derives one absolute target from it, and executes a blocking linear
        move so controller-side trajectory planning completes the point move.
        """
        try:
            offset = np.asarray(delta, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError("delta must contain six numeric pose values") from e
        if offset.shape != (len(self.TCP_AXES),):
            raise ValueError(f"delta must have shape ({len(self.TCP_AXES)},), got {offset.shape}")
        if not np.all(np.isfinite(offset)):
            raise ValueError("delta pose values must all be finite")

        actual_pose = self.get_eef_pose()
        target_pose = actual_pose + offset
        self.set_eef_pose(target_pose)
        return actual_pose, target_pose

    def _get_actual_joint_positions(self) -> dict[str, float]:
        payload = self._check(self.rc.get_actual_joint_position(), return_payload=True)
        if len(payload) != self.N_JOINTS:
            raise JakaError(
                -1,
                f"expected {self.N_JOINTS} joint positions, got {len(payload)}",
                payload=payload,
            )
        try:
            positions = np.asarray(payload, dtype=float)
        except (TypeError, ValueError) as e:
            raise JakaError(-1, "joint positions contain non-numeric values", payload=payload) from e
        if not np.all(np.isfinite(positions)):
            raise JakaError(-1, "joint positions contain non-finite values", payload=payload)
        return {f"{motor}.pos": float(positions[index]) for index, motor in enumerate(self.motors)}

    @staticmethod
    def _normalize_analog_value(value: float, minimum: float, maximum: float, inverted: bool) -> float:
        normalized = float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))
        return 1.0 - normalized if inverted else normalized

    @staticmethod
    def _denormalize_analog_value(position: float, minimum: float, maximum: float, inverted: bool) -> float:
        normalized = 1.0 - position if inverted else position
        return minimum + normalized * (maximum - minimum)

    @check_if_not_connected
    def get_gripper_position(self) -> dict[str, float]:
        """Read normalized gripper feedback from the configured analog input."""
        if not self.config.gripper_analog_input_enabled:
            return {"gripper.pos": float(self.config.gripper_fallback_position)}

        raw_value = self.get_analog_input(
            self.config.gripper_analog_input_iotype,
            self.config.gripper_analog_input_index,
        )
        position = self._normalize_analog_value(
            raw_value,
            self.config.gripper_analog_input_min,
            self.config.gripper_analog_input_max,
            self.config.gripper_analog_input_inverted,
        )
        return {"gripper.pos": position}

    @check_if_not_connected
    def set_gripper_position(self, position: float) -> float:
        """Write a normalized gripper command to the configured analog output."""
        try:
            normalized = float(position)
        except (TypeError, ValueError) as e:
            raise TypeError(f"gripper position {position!r} is not numeric") from e
        if not np.isfinite(normalized):
            raise ValueError("gripper position must be finite")
        normalized = float(np.clip(normalized, 0.0, 1.0))

        if self.config.gripper_analog_output_enabled:
            raw_value = self._denormalize_analog_value(
                normalized,
                self.config.gripper_analog_output_min,
                self.config.gripper_analog_output_max,
                self.config.gripper_analog_output_inverted,
            )
            self.set_analog_output(
                self.config.gripper_analog_output_iotype,
                self.config.gripper_analog_output_index,
                raw_value,
            )
        return normalized

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read measured arm state, status flags, gripper feedback, and cameras."""
        rc = self.rc
        jp = self._check(rc.get_joint_position(), return_payload=True)
        jp_actual = self._check(rc.get_actual_joint_position(), return_payload=True)
        tcp = self._check(rc.get_tcp_position(), return_payload=True)
        tcp_actual = self._check(rc.get_actual_tcp_position(), return_payload=True)

        powered_on = enabled = 0.0
        estop = collision = on_limit = in_pos = drag = 0.0
        rapidrate = 1.0
        try:
            payload = self._check(rc.get_robot_status_simple(), return_payload=True)
            powered_on = float(payload[2])
            enabled = float(payload[3])
        except JakaError:
            pass
        for getter, name in (
            (rc.is_in_estop, "estop"),
            (rc.is_in_collision, "collision"),
            (rc.is_on_limit, "on_limit"),
            (rc.is_in_pos, "in_pos"),
            (rc.is_in_drag_mode, "drag"),
        ):
            try:
                value = float(self._check(getter(), return_payload=True)[0])
            except JakaError:
                value = 0.0
            if name == "estop":
                estop = value
            elif name == "collision":
                collision = value
            elif name == "on_limit":
                on_limit = value
            elif name == "in_pos":
                in_pos = value
            else:
                drag = value
        with suppress(JakaError):
            rapidrate = float(self._check(rc.get_rapidrate(), return_payload=True)[0])

        obs: dict[str, Any] = {}
        for index, motor in enumerate(self.motors):
            obs[f"{motor}.pos"] = float(jp[index])
            obs[f"{motor}.actual_pos"] = float(jp_actual[index])
        for index, axis in enumerate(self.TCP_AXES):
            obs[f"tcp.{axis}"] = float(tcp[index])
            obs[f"tcp.actual_{axis}"] = float(tcp_actual[index])
        actual_eef = np.asarray(tcp_actual, dtype=float).copy()
        actual_eef[:3] /= 1000.0
        for key, value in zip(self.EEF_ACTION_KEYS, actual_eef, strict=True):
            obs[key] = float(value)
        obs.update(self.get_gripper_position())
        obs["status.powered_on"] = powered_on
        obs["status.enabled"] = enabled
        obs["status.estop"] = estop
        obs["status.collision"] = collision
        obs["status.on_limit"] = on_limit
        obs["status.in_pos"] = in_pos
        obs["status.drag_mode"] = drag
        obs["status.rapidrate"] = rapidrate
        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                obs[camera_name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                obs[f"{camera_name}_depth"] = camera.read_latest_depth()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send the active control representation and return the full schema."""
        gripper_position = (
            self.set_gripper_position(action["gripper.pos"])
            if "gripper.pos" in action
            else self.get_gripper_position()["gripper.pos"]
        )

        if self.config.control_mode == "ee_pose":
            applied = self._send_eef_action(action)
            applied.update(self._get_actual_joint_positions())
        else:
            applied = self._send_joint_action(action)
            for key, value in zip(self.EEF_ACTION_KEYS, self.get_eef_pose(), strict=True):
                applied[key] = float(value)
        applied["gripper.pos"] = gripper_position
        return {key: float(applied[key]) for key in self.action_features}

    def _send_joint_action(self, action: RobotAction) -> RobotAction:
        """Drive each named joint toward ``action[m]`` via incremental joint moves.

        Steps:
            1. Snapshot the actual joint positions from the controller.
            2. For every key present in ``action``, clip the goal against
               ``config.max_relative_target`` (per joint, rad). Joints absent
               from ``action`` are left untouched (zero delta).
            3. Issue a non-blocking ``joint_move(INCR)`` of the clipped
               delta at ``config.default_speed_joint``.

        Returns the (clamped) action that was actually sent to the controller
        so the caller can log or compare it against the raw policy output.
        """
        rc = self.rc

        present = self._get_actual_joint_positions()

        # Filter down to the joints the caller is actually driving. Unknown
        # keys are ignored so the caller can pass arbitrary telemetry fields.
        goals: dict[str, float] = {}
        for m in self.motors:
            key = f"{m}.pos"
            if key in action:
                try:
                    goals[key] = float(action[key])
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"JakaRobot.send_action: action[{key!r}]={action[key]!r} is not numeric"
                    ) from e
                if not np.isfinite(goals[key]):
                    raise ValueError(f"JakaRobot.send_action: action[{key!r}] must be finite")

        if goals and self.config.max_relative_target is not None:
            safe = ensure_safe_goal_position(
                {k: (goals[k], present[k]) for k in goals},
                self.config.max_relative_target,
            )
        elif goals:
            safe = dict(goals)
        else:
            safe = {}

        if goals:
            # Build a 6-tuple delta in joint order. Joints the caller didn't
            # touch get a zero delta so the SDK never sees ``None`` entries.
            cmd = tuple(
                safe.get(f"joint_{i + 1}.pos", present[f"joint_{i + 1}.pos"]) - present[f"joint_{i + 1}.pos"]
                for i in range(self.N_JOINTS)
            )
            self._check(rc.joint_move(cmd, INCR, False, self.config.default_speed_joint))

        # Echo back the values actually requested (post-clamp), filling any
        # joints the caller didn't include with their current position so the
        # result contains the active representation; send_action fills the
        # inactive Cartesian representation and gripper field.
        return {**present, **safe}

    def _send_eef_action(self, action: RobotAction) -> RobotAction:
        """Stream one absolute Cartesian Servo Move target in metres/radians."""
        if not self._servo_active:
            raise RuntimeError(
                "JakaRobot ee_pose control requires Servo Move mode; call "
                "servo_enable(True) before send_action()"
            )

        reference = self.get_eef_pose() if self._last_eef_command is None else self._last_eef_command.copy()
        target = reference.copy()

        for index, key in enumerate(self.EEF_ACTION_KEYS):
            if key not in action:
                continue
            try:
                value = float(action[key])
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"JakaRobot.send_action: action[{key!r}]={action[key]!r} is not numeric"
                ) from e
            if not np.isfinite(value):
                raise ValueError(f"JakaRobot.send_action: action[{key!r}] must be finite")
            target[index] = value

        translation_delta = target[:3] - reference[:3]
        translation_norm = float(np.linalg.norm(translation_delta))
        if translation_norm > self.config.max_eef_step_m:
            translation_delta *= self.config.max_eef_step_m / translation_norm
        target[:3] = reference[:3] + translation_delta

        rotation_delta = (target[3:] - reference[3:] + np.pi) % (2.0 * np.pi) - np.pi
        rotation_norm = float(np.linalg.norm(rotation_delta))
        if rotation_norm > self.config.max_eef_step_rad:
            rotation_delta *= self.config.max_eef_step_rad / rotation_norm
        target[3:] = reference[3:] + rotation_delta

        self.servo_eef_frame(target, move_mode=ABS, step_num=self.config.servo_step_num)
        self._last_eef_command = target
        return {key: float(value) for key, value in zip(self.EEF_ACTION_KEYS, target, strict=True)}

    # Direct motion commands bypass the per-step safety clamp.

    @check_if_not_connected
    def move_j(
        self,
        joint_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        is_block: bool = True,
        speed: float | None = None,
        acc: float | None = None,
        tol: float | None = None,
    ) -> None:
        """Joint-space ``joint_move``. ``joint_pos`` in radians."""
        if acc is None:
            self._check(
                self.rc.joint_move(
                    tuple(joint_pos),
                    move_mode,
                    is_block,
                    speed if speed is not None else self.config.default_speed_joint,
                )
            )
            return
        self._check(
            self.rc.joint_move_extend(
                tuple(joint_pos),
                move_mode,
                is_block,
                speed if speed is not None else self.config.default_speed_joint,
                acc,
                tol if tol is not None else 0.1,
            )
        )

    @check_if_not_connected
    def move_l(
        self,
        tcp_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        is_block: bool = True,
        speed: float | None = None,
        acc: float | None = None,
        tol: float | None = None,
        ori_vel: float | None = None,
        ori_acc: float | None = None,
    ) -> None:
        """Cartesian ``linear_move``. ``tcp_pos`` is ``[x, y, z, rx, ry, rz]``
        in mm / rad."""
        linear_speed = speed if speed is not None else self.config.default_speed_linear
        if acc is None:
            self._check(self.rc.linear_move(tuple(tcp_pos), move_mode, is_block, linear_speed))
            return
        # The ``linear_move_extend_ori`` variant lets us tune translational
        # and orientational accels independently; the SDK picks it if the
        # caller opts into orientation limits.
        if ori_vel is not None or ori_acc is not None:
            self._check(
                self.rc.linear_move_extend_ori(
                    tuple(tcp_pos),
                    move_mode,
                    is_block,
                    linear_speed,
                    acc,
                    tol if tol is not None else 0.5,
                    ori_vel if ori_vel is not None else 3.141592653589793,
                    ori_acc if ori_acc is not None else 12.566370614359172,
                )
            )
            return
        self._check(
            self.rc.linear_move_extend(
                tuple(tcp_pos),
                move_mode,
                is_block,
                linear_speed,
                acc,
                tol if tol is not None else 0.5,
            )
        )

    @check_if_not_connected
    def move_c(
        self,
        end_pos: list[float] | tuple[float, ...],
        mid_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        is_block: bool = True,
        speed: float | None = None,
        acc: float | None = None,
        tol: float | None = None,
        circle_cnt: int = 1,
        circle_mode: int | None = None,
    ) -> None:
        """Circular ``circular_move`` via a mid-point pose (mm / rad)."""
        linear_speed = speed if speed is not None else self.config.default_speed_linear
        linear_acc = acc if acc is not None else 1.0
        linear_tol = tol if tol is not None else 0.5
        if circle_mode is None:
            self._check(
                self.rc.circular_move_extend(
                    tuple(end_pos),
                    tuple(mid_pos),
                    move_mode,
                    is_block,
                    linear_speed,
                    linear_acc,
                    linear_tol,
                    None,
                    int(circle_cnt),
                )
            )
            return
        self._check(
            self.rc.circular_move_extend_mode(
                tuple(end_pos),
                tuple(mid_pos),
                move_mode,
                is_block,
                linear_speed,
                linear_acc,
                linear_tol,
                None,
                int(circle_cnt),
                int(circle_mode),
            )
        )

    @check_if_not_connected
    def jog(
        self,
        aj_num: int,
        jog_vel: float,
        move_mode: int = CONT,
        coord_type: int = COORD_BASE,
        pos_cmd: float = 0.0,
    ) -> None:
        """Manual-mode ``jog``. With ``move_mode = CONT`` motion continues
        until :meth:`jog_stop` is called."""
        self._check(self.rc.jog(int(aj_num), int(move_mode), int(coord_type), float(jog_vel), float(pos_cmd)))

    @check_if_not_connected
    def jog_stop(self, joint_num: int = -1) -> None:
        """Halt one (``0..5``) or all (``-1``) jogging axes."""
        self._check(self.rc.jog_stop(int(joint_num)))

    @check_if_not_connected
    def motion_abort(self) -> None:
        """Force-stop the currently executing motion."""
        self._check(self.rc.motion_abort())

    @check_if_not_connected
    def is_in_pos(self) -> bool:
        """Return True if the arm has settled at its target."""
        return bool(self._check(self.rc.is_in_pos(), return_payload=True)[0])

    @check_if_not_connected
    def get_motion_status(self) -> dict[str, Any]:
        """Snapshot of the controller's motion-queue telemetry."""
        payload = self._check(self.rc.get_motion_status(), return_payload=True)
        keys = (
            "motion_line",
            "motion_line_sdk",
            "inpos",
            "err_add_line",
            "queue",
            "active_queue",
            "queue_full",
            "paused",
            "on_limit",
            "in_estop",
            "in_collision",
        )
        return {k: payload[i] for i, k in enumerate(keys) if i < len(payload)}

    # ── Power / enable / drag ──

    def power_on(self) -> None:
        """Power on the arm (~8 s delay)."""
        self._check(self.rc.power_on())
        logger.info("JakaRobot: powered on")

    @check_if_not_connected
    def power_off(self) -> None:
        """Power off the arm."""
        self._check(self.rc.power_off())

    def enable_robot(self) -> None:
        """Engage the servos."""
        self._check(self.rc.enable_robot())
        logger.info("JakaRobot: enabled")

    @check_if_not_connected
    def disable_robot(self) -> None:
        """Disengage the servos."""
        self._check(self.rc.disable_robot())

    @check_if_not_connected
    def shutdown(self) -> None:
        """Shut down the controller cabinet."""
        self._check(self.rc.shut_down())

    @check_if_not_connected
    def set_drag_mode(self, enable: bool) -> None:
        """Toggle gravity-compensated hand-guidance (drag) mode."""
        self._check(self.rc.drag_mode_enable(bool(enable)))

    @check_if_not_connected
    def is_in_drag_mode(self) -> bool:
        return bool(self._check(self.rc.is_in_drag_mode(), return_payload=True)[0])

    @check_if_not_connected
    def set_rapidrate(self, rate: float) -> None:
        """Set the global speed multiplier in ``[0.0, 1.0]``."""
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"rapidrate must be in [0, 1], got {rate}")
        self._check(self.rc.set_rapidrate(float(rate)))

    @check_if_not_connected
    def get_rapidrate(self) -> float:
        return float(self._check(self.rc.get_rapidrate(), return_payload=True)[0])

    @check_if_not_connected
    def set_motion_planner(self, planner_type: int = PLANNER_T) -> None:
        """Switch motion planner: ``PLANNER_DISABLED`` (``-1``), ``PLANNER_T``
        (``0``), ``PLANNER_S`` (``1``)."""
        if planner_type not in (PLANNER_DISABLED, PLANNER_T, PLANNER_S):
            raise ValueError(f"planner_type must be one of {PLANNER_DISABLED, PLANNER_T, PLANNER_S}")
        self._check(self.rc.set_motion_planner(int(planner_type)))

    # ── Tool / user-frame ──

    @check_if_not_connected
    def set_tool_id(self, id: int) -> None:
        if not 0 <= int(id) <= 15:
            raise ValueError("tool id must be in [0, 15]")
        self._check(self.rc.set_tool_id(int(id)))

    @check_if_not_connected
    def get_tool_id(self) -> int:
        return int(self._check(self.rc.get_tool_id(), return_payload=True)[0])

    @check_if_not_connected
    def set_user_frame_id(self, id: int) -> None:
        if not 0 <= int(id) <= 15:
            raise ValueError("user frame id must be in [0, 15]")
        self._check(self.rc.set_user_frame_id(int(id)))

    @check_if_not_connected
    def get_user_frame_id(self) -> int:
        return int(self._check(self.rc.get_user_frame_id(), return_payload=True)[0])

    @check_if_not_connected
    def set_tool_data(self, id: int, tcp: list[float], name: str = "") -> None:
        """Define a tool's TCP offset ``[x, y, z, rx, ry, rz]`` (mm / rad)."""
        self._check(self.rc.set_tool_data(int(id), tuple(tcp), name))

    @check_if_not_connected
    def get_tool_data(self, id: int) -> tuple[int, tuple[float, ...]]:
        payload = self._check(self.rc.get_tool_data(int(id)), return_payload=True)
        return int(payload[0]), tuple(payload[1])

    @check_if_not_connected
    def set_user_frame_data(self, id: int, frame: list[float], name: str = "") -> None:
        """Define a user-frame origin ``[x, y, z, rx, ry, rz]`` (mm / rad)."""
        self._check(self.rc.set_user_frame_data(int(id), tuple(frame), name))

    @check_if_not_connected
    def get_user_frame_data(self, id: int) -> tuple[int, tuple[float, ...]]:
        payload = self._check(self.rc.get_user_frame_data(int(id)), return_payload=True)
        return int(payload[0]), tuple(payload[1])

    # ── Safety / error utilities ──

    @check_if_not_connected
    def is_estopped(self) -> bool:
        return bool(self._check(self.rc.is_in_estop(), return_payload=True)[0])

    @check_if_not_connected
    def is_on_limit(self) -> bool:
        return bool(self._check(self.rc.is_on_limit(), return_payload=True)[0])

    @check_if_not_connected
    def is_in_collision(self) -> bool:
        return bool(self._check(self.rc.is_in_collision(), return_payload=True)[0])

    @check_if_not_connected
    def recover_collision(self) -> None:
        """Exit collision-protection mode after clearing the cause."""
        self._check(self.rc.collision_recover())

    @check_if_not_connected
    def clear_error(self) -> None:
        """Reset controller-side error state."""
        self._check(self.rc.clear_error())

    @check_if_not_connected
    def set_collision_level(self, level: int) -> None:
        if not 0 <= int(level) <= 5:
            raise ValueError("collision level must be in [0, 5]")
        self._check(self.rc.set_collision_level(int(level)))

    @check_if_not_connected
    def get_collision_level(self) -> int:
        return int(self._check(self.rc.get_collision_level(), return_payload=True)[0])

    @check_if_not_connected
    def get_last_error(self) -> int:
        return int(self._check(self.rc.get_last_error(), return_payload=True)[0])

    # ── Servo-mode (fast closed-loop) commands ──

    @staticmethod
    def _validate_servo_step_num(step_num: int) -> int:
        step = int(step_num)
        if not 1 <= step <= DEFAULT_SERVO_STEP_NUM:
            raise ValueError(
                f"step_num must be in [1, {DEFAULT_SERVO_STEP_NUM}] to maintain "
                "a nominal rate of at least 30 Hz"
            )
        return step

    @staticmethod
    def _validate_servo_target(target, name: str) -> np.ndarray:
        try:
            values = np.asarray(target, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name} must contain six numeric values") from e
        if values.shape != (6,):
            raise ValueError(f"{name} must have shape (6,), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} values must all be finite")
        return values

    def servo_frame_period_s(self, step_num: int | None = None) -> float:
        """Return the nominal controller period for one Servo Move frame."""
        step = self._validate_servo_step_num(self.config.servo_step_num if step_num is None else step_num)
        return step * SERVO_CYCLE_S

    @contextmanager
    @check_if_not_connected
    def servo_stream(self, step_num: int | None = None) -> Generator[float]:
        """Enter Servo Move mode and yield its nominal frame period in seconds.

        The caller must plan a smooth trajectory and continuously send frames;
        this synchronous context manager does not run a loop or repeat targets.
        Actual frequency depends on caller timing, SDK latency, and the network.
        """
        if self._servo_active:
            raise RuntimeError("a JAKA Servo Move stream is already active")
        period_s = self.servo_frame_period_s(step_num)
        self._check(self.rc.servo_move_enable(True, True))
        self._servo_active = True
        self._servo_queue_warned = False
        self._last_eef_command = None
        try:
            yield period_s
        finally:
            self._servo_active = False
            self._last_eef_command = None
            try:
                self._check(self.rc.servo_move_enable(False, True))
            except JakaError as e:
                logger.warning("JakaRobot: could not exit Servo Move mode: %s", e)

    @check_if_not_connected
    def servo_enable(self, enable: bool = True) -> None:
        """Toggle raw Servo Move mode; prefer :meth:`servo_stream` for cleanup."""
        self._check(self.rc.servo_move_enable(bool(enable), True))
        self._servo_active = bool(enable)
        self._last_eef_command = None
        if enable:
            self._servo_queue_warned = False

    @check_if_not_connected
    def is_in_servo(self) -> bool:
        return bool(self._check(self.rc.is_in_servomove(), return_payload=True)[0])

    def _servo_queue_depth(self, ret) -> int | None:
        payload = self._check(ret, return_payload=True)
        if not payload:
            return None
        try:
            queue_depth = int(payload[0])
        except (TypeError, ValueError) as e:
            raise JakaError(-1, "invalid Servo Move queue depth response", payload=payload) from e
        if not 0 <= queue_depth <= SERVO_QUEUE_MAX:
            raise JakaError(-1, f"invalid Servo Move queue depth: {queue_depth}", payload=payload)
        if queue_depth >= self.config.servo_queue_warn_depth and not self._servo_queue_warned:
            logger.warning(
                "JakaRobot: Servo Move queue depth %d/%d; actual send rate may be too high",
                queue_depth,
                SERVO_QUEUE_MAX,
            )
            self._servo_queue_warned = True
        return queue_depth

    @check_if_not_connected
    def servo_j(
        self,
        joint_pos: list[float] | tuple[float, ...] | np.ndarray,
        move_mode: int = ABS,
        step_num: int | None = None,
    ) -> int | None:
        """Push one joint Servo Move frame in radians.

        The caller must continuously call this method at the nominal period;
        one frame alone has no useful effect. Both absolute and incremental
        targets are supported.
        """
        if not self._servo_active:
            raise RuntimeError("enter servo_stream() or call servo_enable(True) before servo_j()")
        if move_mode not in (ABS, INCR):
            raise ValueError(f"move_mode must be ABS ({ABS}) or INCR ({INCR})")
        step = self._validate_servo_step_num(self.config.servo_step_num if step_num is None else step_num)
        target = self._validate_servo_target(joint_pos, "joint_pos")
        return self._servo_queue_depth(self.rc.servo_j(tuple(target), int(move_mode), step))

    @check_if_not_connected
    def servo_p(
        self,
        tcp_pos: list[float] | tuple[float, ...] | np.ndarray,
        move_mode: int = ABS,
        step_num: int | None = None,
    ) -> int | None:
        """Push one raw Cartesian Servo Move frame in SDK mm/rad units."""
        if not self._servo_active:
            raise RuntimeError("enter servo_stream() or call servo_enable(True) before servo_p()")
        if move_mode not in (ABS, INCR):
            raise ValueError(f"move_mode must be ABS ({ABS}) or INCR ({INCR})")
        step = self._validate_servo_step_num(self.config.servo_step_num if step_num is None else step_num)
        target = self._validate_servo_target(tcp_pos, "tcp_pos")
        return self._servo_queue_depth(self.rc.servo_p(tuple(target), int(move_mode), step))

    def servo_joint_frame(
        self,
        joint_pos: list[float] | tuple[float, ...] | np.ndarray,
        move_mode: int = ABS,
        step_num: int | None = None,
    ) -> int | None:
        """Push one validated high-frequency joint target in radians."""
        return self.servo_j(joint_pos, move_mode=move_mode, step_num=step_num)

    def servo_eef_frame(
        self,
        pose: list[float] | tuple[float, ...] | np.ndarray,
        move_mode: int = ABS,
        step_num: int | None = None,
    ) -> int | None:
        """Push ``[x, y, z, rx, ry, rz]`` in metres/radians.

        In incremental mode, XYZ values are per-frame metre deltas. The caller
        must continuously stream a smooth trajectory at the nominal period.
        """
        target = self._validate_servo_target(pose, "pose").copy()
        target[:3] *= 1000.0
        return self.servo_p(target, move_mode=move_mode, step_num=step_num)

    # ── IO helpers (work on cabinet / tool / extended IO blocks) ──

    @staticmethod
    def _io_kind() -> dict[str, int]:
        return {"cabinet": IO_CABINET, "tool": IO_TOOL, "extend": IO_EXTEND}

    @check_if_not_connected
    def get_digital_input(self, iotype: int, index: int) -> int:
        return int(self._check(self.rc.get_digital_input(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def get_digital_output(self, iotype: int, index: int) -> int:
        return int(self._check(self.rc.get_digital_output(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def set_digital_output(self, iotype: int, index: int, value: bool | int) -> None:
        self._check(self.rc.set_digital_output(int(iotype), int(index), bool(value)))

    @check_if_not_connected
    def get_analog_input(self, iotype: int, index: int) -> float:
        return float(self._check(self.rc.get_analog_input(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def get_analog_output(self, iotype: int, index: int) -> float:
        return float(self._check(self.rc.get_analog_output(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def set_analog_output(self, iotype: int, index: int, value: float) -> None:
        self._check(self.rc.set_analog_output(int(iotype), int(index), float(value)))

    # ── App-script (JAKA ``*.jks`` program) interface ──

    @check_if_not_connected
    def program_load(self, name: str) -> None:
        self._check(self.rc.program_load(name))

    @check_if_not_connected
    def program_run(self) -> None:
        self._check(self.rc.program_run())

    @check_if_not_connected
    def program_pause(self) -> None:
        self._check(self.rc.program_pause())

    @check_if_not_connected
    def program_resume(self) -> None:
        self._check(self.rc.program_resume())

    @check_if_not_connected
    def program_abort(self) -> None:
        self._check(self.rc.program_abort())

    @check_if_not_connected
    def get_loaded_program(self) -> str:
        return str(self._check(self.rc.get_loaded_program(), return_payload=True)[0])

    @check_if_not_connected
    def get_program_state(self) -> int:
        """Return one of ``PROG_STOPPED`` (0), ``PROG_RUNNING`` (1), ``PROG_PAUSED`` (2)."""
        return int(self._check(self.rc.get_program_state(), return_payload=True)[0])

    # ── Misc ──

    @check_if_not_connected
    def get_sdk_version(self) -> str:
        """Return the SDK version string (e.g. ``"1.7.2"``)."""
        payload = self._check(self.rc.get_sdk_version(), return_payload=True)
        version = payload[0]
        if isinstance(version, (tuple, list)):
            return ".".join(str(x) for x in version)
        return str(version)

    @check_if_not_connected
    def get_controller_ip(self) -> list[str]:
        """Return controller IPs reachable from this client."""
        payload = self._check(self.rc.get_controller_ip(), return_payload=True)
        ip_list = payload[0]
        return list(ip_list) if isinstance(ip_list, (tuple, list)) else [str(ip_list)]

    @property
    def controller_ip(self) -> str:
        return self.config.ip

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return f"JakaRobot(ip={self.config.ip!r}, {state})"


__all__ = [
    "ABS",
    "CART_RX",
    "CART_RY",
    "CART_RZ",
    "CART_X",
    "CART_Y",
    "CART_Z",
    "CONT",
    "COORD_BASE",
    "COORD_JOINT",
    "COORD_TOOL",
    "DEFAULT_SERVO_STEP_NUM",
    "INCR",
    "IO_CABINET",
    "IO_EXTEND",
    "IO_TOOL",
    "JAKA_ERROR_MESSAGES",
    "PLANNER_DISABLED",
    "PLANNER_S",
    "PLANNER_T",
    "PROG_PAUSED",
    "PROG_RUNNING",
    "PROG_STOPPED",
    "SERVO_CYCLE_S",
    "SERVO_QUEUE_MAX",
    "JakaError",
    "JakaRobot",
    "JakaRobotConfig",
]


if __name__ == "__main__":
    import select
    import time

    logging.basicConfig(level=logging.INFO)

    offset_m = 0.05
    cfg = JakaRobotConfig(ip="192.168.1.31", auto_enable=True)
    robot = JakaRobot(cfg)
    print(robot)
    with robot:
        print("SDK:", robot.get_sdk_version())
        initial_pose = robot.get_eef_pose()
        print(f"initial EEF pose (m/rad): {np.round(initial_pose, 4)}")

        choice = (
            input("Type MOVE for the waypoint test, KEYBOARD for keyboard teleop, anything else to quit: ")
            .strip()
            .upper()
        )

        if choice == "MOVE":
            print(
                "Motion path: origin -> up -> origin -> down -> origin -> left -> origin -> right -> origin"
            )
            print(f"Translation offset: {offset_m:.3f} m; linear speed: {cfg.default_speed_linear:.1f} mm/s")

            waypoints = (
                ("up", np.array([0.0, 0.0, offset_m, 0.0, 0.0, 0.0])),
                ("origin", np.zeros(6)),
                ("down", np.array([0.0, 0.0, -offset_m, 0.0, 0.0, 0.0])),
                ("origin", np.zeros(6)),
                ("left", np.array([0.0, offset_m, 0.0, 0.0, 0.0, 0.0])),
                ("origin", np.zeros(6)),
                ("right", np.array([0.0, -offset_m, 0.0, 0.0, 0.0, 0.0])),
                ("origin", np.zeros(6)),
            )

            try:
                for name, offset in waypoints:
                    target = initial_pose + offset
                    print(f"moving {name}: {np.round(target, 4)}")
                    robot.set_eef_pose(target)
                    actual = robot.get_eef_pose()
                    print(f"reached {name}: {np.round(actual, 4)}")
            except (JakaError, KeyboardInterrupt) as e:
                print(f"Motion test interrupted: {e}")
                try:
                    robot.motion_abort()
                    print("Attempting to return to the initial pose...")
                    robot.set_eef_pose(initial_pose)
                except (JakaError, KeyboardInterrupt) as return_error:
                    print(f"Could not return to the initial pose: {return_error}")
                raise

            print("Motion test complete; EEF returned to the initial pose.")

        elif choice == "KEYBOARD":
            try:
                import termios
                import tty
            except ImportError as e:
                raise SystemExit("Terminal raw mode requires POSIX termios/tty (Linux/macOS).") from e

            # One key event moves the TCP by this amount.
            KEY_STEP_M = 0.005

            def parse_key(buf: bytes) -> str | None:
                """Decode a complete sequence read from a cbreak terminal."""
                if not buf:
                    return None
                if buf == b"\x1b":
                    return "esc"
                if buf.startswith(b"\x1b[") and len(buf) >= 3:
                    return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                        buf[2:3].decode("ascii", "ignore")
                    )
                if buf.startswith(b"\x1bO") and len(buf) >= 3:
                    return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                        buf[2:3].decode("ascii", "ignore")
                    )
                ch = buf[:1].decode("ascii", "ignore").lower()
                if ch in ("w", "a", "s", "d", "z", "c", "q"):
                    return ch
                return None

            def read_key(fd: int) -> str | None:
                """Non-blocking read of one key, returning its decoded name or None."""
                r, _, _ = select.select([fd], [], [], 0)
                if not r:
                    return None
                first = os.read(fd, 1)
                if first == b"\x1b":
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if not r:
                        return "esc"
                    second = os.read(fd, 1)
                    if second == b"[" or second == b"O":
                        r, _, _ = select.select([fd], [], [], 0.05)
                        if not r:
                            return "esc"
                        third = os.read(fd, 1)
                        return parse_key(b"\x1b" + second + third)
                    return parse_key(b"\x1b" + second)
                return parse_key(first)

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            print(
                "Keyboard teleop:\n"
                "  arrows / wasd:  X/Y plane (base frame)\n"
                "  z / c:          +Z / -Z\n"
                "  q or ESC:       quit"
            )
            print(f"Step: {KEY_STEP_M * 1000:.1f} mm per key event")
            print(f"Initial EEF pose (m/rad): {np.round(initial_pose, 4)}")

            quit_requested = False
            moves_sent = 0
            target_pose = initial_pose.copy()
            print(f"target (m/rad): {np.round(target_pose, 4)}")
            try:
                while not quit_requested:
                    key = read_key(fd)
                    if key is None:
                        time.sleep(0.01)
                        continue
                    if key in ("q", "esc"):
                        quit_requested = True
                        break

                    delta = np.zeros(6)
                    if key == "up" or key == "w":
                        delta[1] = -KEY_STEP_M
                    elif key == "down" or key == "s":
                        delta[1] = KEY_STEP_M
                    elif key == "left" or key == "a":
                        delta[0] = -KEY_STEP_M
                    elif key == "right" or key == "d":
                        delta[0] = KEY_STEP_M
                    elif key == "z":
                        delta[2] = KEY_STEP_M
                    elif key == "c":
                        delta[2] = -KEY_STEP_M

                    if np.any(delta != 0):
                        actual_pose, target_pose = robot.move_eef_relative(delta)
                        moves_sent += 1
                        print(
                            f"key={key} actual(m/rad)={np.round(actual_pose, 4)} "
                            f"delta(m)={np.round(delta, 4)} target(m/rad)={np.round(target_pose, 4)}"
                        )
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except termios.error:
                    print(
                        "Warning: could not restore terminal settings; you may need to reset your terminal."
                    )
                final_pose = robot.get_eef_pose()
                print(f"Moves sent: {moves_sent}")
                print(f"Final EEF pose (m/rad): {np.round(final_pose, 4)}")
                print(f"Final target    (m/rad): {np.round(target_pose, 4)}")
                if quit_requested:
                    print("Keyboard teleop ended by user request.")
        else:
            print("No interaction selected; exiting.")
