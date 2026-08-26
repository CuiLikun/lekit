"""LeRobot adapter for AgileX Piper arms driven by ``pyAgxArm``.

The public interface uses SI units throughout: joint positions are radians and
the optional AGX gripper opening is metres.  Vendor objects are created lazily
in :meth:`PiperRobot.connect`, so importing this module does not require the
optional hardware dependency.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, Literal

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)


class PiperFeedbackError(RuntimeError):
    """Raised when Piper feedback is absent or does not match the adapter contract."""


class PiperCameraTimeoutError(TimeoutError):
    """A camera timeout carrying the arm feedback already read for this frame."""

    def __init__(self, camera_name: str, observation: RobotObservation, cause: TimeoutError):
        self.camera_name = camera_name
        self.observation = dict(observation)
        super().__init__(f"Piper camera {camera_name!r} timed out: {cause}")


def create_piper_sdk_arm(config: dict[str, Any]) -> Any:
    """Create a ``pyAgxArm`` Piper driver at the optional-dependency seam."""

    try:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
    except ImportError as exc:
        raise ImportError(
            "PiperRobot requires pyAgxArm. Install this project with the 'piper' extra."
        ) from exc

    sdk_config = create_agx_arm_config(**config)
    return AgxArmFactory.create_arm(sdk_config)


def _has_complete_joint_feedback(arm: Any) -> bool:
    """Require all three Piper joint-pair CAN frames from the pinned SDK parser."""

    parser = getattr(arm, "_parser", None)
    return parser is not None and all(
        getattr(parser, frame_name, None) is not None for frame_name in ("joint_12", "joint_34", "joint_56")
    )


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    """Connection, safety, gripper, and camera settings for :class:`PiperRobot`."""

    channel: str = "can0"
    interface: Literal["socketcan", "agx_cando", "slcan"] = "socketcan"
    bitrate: int = 1_000_000
    robot_model: Literal["piper", "piper_h", "piper_l", "piper_x"] = "piper"
    firmware_version: Literal["default", "v183", "v188", "v189"] = "default"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    include_gripper: bool = True
    gripper_force_n: float = 1.0
    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float = 0.07

    auto_enable: bool = True
    feedback_timeout_s: float = 2.0
    feedback_poll_interval_s: float = 0.01
    enable_timeout_s: float = 5.0
    enable_poll_interval_s: float = 0.01
    speed_percent: int = 20
    disable_on_disconnect: bool = False

    max_relative_target: float | dict[str, float] | None = 0.05
    joint_limit_tolerance_rad: float = 0.02

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.channel.strip():
            raise ValueError("PiperRobotConfig.channel must not be empty.")
        if self.bitrate <= 0:
            raise ValueError("PiperRobotConfig.bitrate must be positive.")
        if not 0 <= self.speed_percent <= 100:
            raise ValueError("PiperRobotConfig.speed_percent must be in [0, 100].")
        if not math.isfinite(self.enable_timeout_s) or self.enable_timeout_s <= 0:
            raise ValueError("PiperRobotConfig.enable_timeout_s must be positive and finite.")
        if not math.isfinite(self.enable_poll_interval_s) or self.enable_poll_interval_s <= 0:
            raise ValueError("PiperRobotConfig.enable_poll_interval_s must be positive and finite.")
        if not math.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0:
            raise ValueError("PiperRobotConfig.feedback_timeout_s must be positive and finite.")
        if not math.isfinite(self.feedback_poll_interval_s) or self.feedback_poll_interval_s <= 0:
            raise ValueError("PiperRobotConfig.feedback_poll_interval_s must be positive and finite.")
        if not math.isfinite(self.gripper_force_n) or not 0.0 <= self.gripper_force_n <= 3.0:
            raise ValueError("PiperRobotConfig.gripper_force_n must be in [0, 3].")
        if (
            not math.isfinite(self.gripper_min_width_m)
            or not math.isfinite(self.gripper_max_width_m)
            or self.gripper_min_width_m < 0.0
            or self.gripper_min_width_m >= self.gripper_max_width_m
        ):
            raise ValueError("Piper gripper width range must be finite, non-negative, and increasing.")
        if not math.isfinite(self.joint_limit_tolerance_rad) or self.joint_limit_tolerance_rad < 0:
            raise ValueError("PiperRobotConfig.joint_limit_tolerance_rad must be finite and non-negative.")
        self.max_relative_target = self._validate_max_relative_target(self.max_relative_target)

    @staticmethod
    def _validate_max_relative_target(
        value: float | dict[str, float] | None,
    ) -> float | dict[str, float] | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("max_relative_target must be positive or None.")
        if isinstance(value, (int, float)):
            result = float(value)
            if not math.isfinite(result) or result <= 0:
                raise ValueError("max_relative_target must be positive and finite.")
            return result
        if isinstance(value, dict):
            expected = {f"joint_{index}.pos" for index in range(1, 7)}
            if set(value) != expected:
                raise ValueError("max_relative_target mapping must contain joint_1.pos through joint_6.pos.")
            result = {key: float(limit) for key, limit in value.items()}
            if any(not math.isfinite(limit) or limit <= 0 for limit in result.values()):
                raise ValueError("max_relative_target mapping values must be positive and finite.")
            return result
        raise ValueError("max_relative_target must be a positive number, a joint mapping, or None.")


class PiperRobot(Robot):
    """A standard LeRobot robot backed by the ``pyAgxArm`` Piper driver."""

    config_class: ClassVar[type] = PiperRobotConfig
    name: ClassVar[str] = "piper"
    motors: ClassVar[tuple[str, ...]] = tuple(f"joint_{index}" for index in range(1, 7))
    _JOINT_KEYS: ClassVar[tuple[str, ...]] = tuple(f"{motor}.pos" for motor in motors)

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}
        self.arm: Any | None = None
        self.gripper: Any | None = None
        self._joint_limits: dict[str, tuple[float, float]] | None = None

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = dict.fromkeys(self._JOINT_KEYS, float)
        if self.config.include_gripper:
            features["gripper.pos"] = float
        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{name}_depth"] = (camera.height, camera.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        features: dict[str, Any] = dict.fromkeys(self._JOINT_KEYS, float)
        if self.config.include_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        if self.arm is None:
            return False
        try:
            controller_connected = bool(self.arm.is_connected())
        except Exception:  # nosec B110 - a broken SDK handle is disconnected
            return False
        return controller_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """No-op: Piper joint calibration is managed by its controller."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Create the SDK driver, connect feedback, configure, and optionally enable."""

        del calibrate
        sdk_args = {
            "robot": self.config.robot_model,
            "firmeware_version": self.config.firmware_version,
            "channel": self.config.channel,
            "interface": self.config.interface,
            "bitrate": self.config.bitrate,
        }
        arm = create_piper_sdk_arm(sdk_args)
        self.arm = arm
        try:
            if self.config.include_gripper:
                self.gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            arm.connect()
            self._wait_for_initial_feedback()
            self.configure()
            if self.config.auto_enable:
                self._enable_arm()
            for camera in self.cameras.values():
                camera.connect()
        except BaseException:
            self.disconnect()
            raise
        logger.info("Piper arm on %s connected.", self.config.channel)

    def _wait_for_initial_feedback(self) -> None:
        """Do not report a connected robot until its first complete state is readable."""

        deadline = time.monotonic() + self.config.feedback_timeout_s
        joints_ready = False
        gripper_ready = not self.config.include_gripper
        last_error: PiperFeedbackError | None = None
        while not joints_ready or not gripper_ready:
            if not joints_ready:
                try:
                    self._read_joint_positions()
                except PiperFeedbackError as exc:
                    last_error = exc
                else:
                    if _has_complete_joint_feedback(self.arm):
                        joints_ready = True
                    else:
                        last_error = PiperFeedbackError(
                            "Piper joint feedback is incomplete; waiting for joint pairs 1-2, 3-4, and 5-6."
                        )
            if not gripper_ready:
                try:
                    self._read_gripper_width()
                except PiperFeedbackError as exc:
                    last_error = exc
                else:
                    gripper_ready = True
            if joints_ready and gripper_ready:
                return
            if time.monotonic() >= deadline:
                detail = f" Last feedback error: {last_error}" if last_error else ""
                raise TimeoutError(
                    f"Piper initial feedback timed out after {self.config.feedback_timeout_s:g} seconds."
                    + detail
                )
            time.sleep(self.config.feedback_poll_interval_s)

    def configure(self) -> None:
        """Enable model limits and apply the global motion speed percentage."""

        if self.arm is not None:
            self.arm.set_joint_limits_enabled(True)
            if not self.arm.get_joint_limits_enabled():
                raise RuntimeError("pyAgxArm did not enable Piper model joint limits.")
            self._joint_limits = self._read_model_joint_limits()
            self.arm.set_speed_percent(self.config.speed_percent)

    def _read_model_joint_limits(self) -> dict[str, tuple[float, float]]:
        assert self.arm is not None
        raw_limits = self.arm.get_config().get("joint_limits")
        if not isinstance(raw_limits, dict):
            raise RuntimeError("pyAgxArm did not provide Piper model joint limits.")
        expected = {f"joint{index}" for index in range(1, 7)}
        if set(raw_limits) != expected:
            raise RuntimeError("pyAgxArm Piper model joint limits are incomplete.")
        limits: dict[str, tuple[float, float]] = {}
        for index, key in enumerate(self._JOINT_KEYS, start=1):
            bounds = raw_limits[f"joint{index}"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise RuntimeError(f"pyAgxArm joint{index} limits are malformed.")
            lower, upper = map(float, bounds)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise RuntimeError(f"pyAgxArm joint{index} limits are malformed.")
            limits[key] = (lower, upper)
        return limits

    def _enable_arm(self) -> None:
        assert self.arm is not None
        deadline = time.monotonic() + self.config.enable_timeout_s
        while not self.arm.enable():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Piper arm enable timed out after {self.config.enable_timeout_s:g} seconds."
                )
            time.sleep(self.config.enable_poll_interval_s)

    def disconnect(self) -> None:
        """Release cameras and CAN resources without dropping the arm by default."""

        arm = self.arm
        if arm is None:
            return
        try:
            try:
                for camera in self.cameras.values():
                    try:
                        camera.disconnect()
                    except Exception as exc:  # nosec B110 - continue hardware cleanup
                        logger.warning("Piper camera cleanup failed: %s", exc)
                if self.config.disable_on_disconnect:
                    arm.disable()
            finally:
                arm.disconnect()
        finally:
            self.arm = None
            self.gripper = None
            self._joint_limits = None

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read joint angles, optional gripper width, and configured cameras."""

        observation: dict[str, Any] = self._read_joint_positions()
        if self.config.include_gripper:
            observation["gripper.pos"] = self._read_gripper_width()
        for name, camera in self.cameras.items():
            try:
                if getattr(camera, "use_rgb", True):
                    observation[name] = camera.read_latest()
                if getattr(camera, "use_depth", False):
                    observation[f"{name}_depth"] = camera.read_latest_depth()
            except TimeoutError as exc:
                raise PiperCameraTimeoutError(name, observation, exc) from exc
        return observation

    def _read_joint_positions(self) -> dict[str, float]:
        assert self.arm is not None
        feedback = self.arm.get_joint_angles()
        values = getattr(feedback, "msg", None)
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise PiperFeedbackError("Piper joint angles feedback is unavailable or malformed.")
        try:
            joints = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise PiperFeedbackError("Piper joint angles feedback contains non-numeric values.") from exc
        if any(not math.isfinite(value) for value in joints):
            raise PiperFeedbackError("Piper joint angles feedback contains non-finite values.")
        return dict(zip(self._JOINT_KEYS, joints, strict=True))

    def _read_gripper_width(self) -> float:
        if self.gripper is None:
            raise PiperFeedbackError("Piper AGX gripper is not initialized.")
        feedback = self.gripper.get_gripper_status()
        message = getattr(feedback, "msg", None)
        if message is None:
            raise PiperFeedbackError("Piper gripper feedback is unavailable.")
        if getattr(message, "mode", None) != "width":
            raise PiperFeedbackError("Piper gripper feedback must be in width mode.")
        try:
            width = float(message.value)
        except (AttributeError, TypeError, ValueError) as exc:
            raise PiperFeedbackError("Piper gripper width feedback is malformed.") from exc
        if not math.isfinite(width):
            raise PiperFeedbackError("Piper gripper width feedback is non-finite.")
        return width

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Apply a partial, safety-bounded joint/gripper position action."""

        unknown = set(action) - set(self.action_features)
        if unknown:
            raise ValueError(f"Unsupported Piper action fields: {sorted(unknown)}")
        validated_action = {key: self._finite_action_value(key, value) for key, value in action.items()}
        current = self._read_joint_positions()
        requested: dict[str, float] = {}
        for key in self._JOINT_KEYS:
            if key not in validated_action:
                continue
            requested[key] = validated_action[key]

        safe = requested
        if requested:
            self._validate_current_joint_positions(current)
        if requested and self.config.max_relative_target is not None:
            limits = self.config.max_relative_target
            if isinstance(limits, dict):
                limits = {key: limits[key] for key in requested}
            safe = ensure_safe_goal_position(
                {key: (goal, current[key]) for key, goal in requested.items()},
                limits,
            )

        applied: dict[str, float] = dict(current)
        applied.update(safe)
        if requested:
            assert self.arm is not None
            if self._joint_limits is None:
                raise RuntimeError("Piper model joint limits are unavailable.")
            applied = {
                key: self._quantize_joint_target(
                    min(max(value, self._joint_limits[key][0]), self._joint_limits[key][1])
                )
                for key, value in applied.items()
            }
            self.arm.move_j([applied[key] for key in self._JOINT_KEYS])

        if self.config.include_gripper:
            if "gripper.pos" in validated_action:
                width = validated_action["gripper.pos"]
                width = min(max(width, self.config.gripper_min_width_m), self.config.gripper_max_width_m)
                assert self.gripper is not None
                self.gripper.move_gripper_m(value=width, force=self.config.gripper_force_n)
            else:
                width = self._read_gripper_width()
            applied["gripper.pos"] = width
        return applied

    def _validate_current_joint_positions(self, current: dict[str, float]) -> None:
        if self._joint_limits is None:
            raise RuntimeError("Piper model joint limits are unavailable.")
        tolerance = self.config.joint_limit_tolerance_rad
        for key, value in current.items():
            lower, upper = self._joint_limits[key]
            if value < lower - tolerance or value > upper + tolerance:
                raise PiperFeedbackError(
                    f"Piper measured {key}={value:.6f} rad outside model limits "
                    f"[{lower:.6f}, {upper:.6f}] with {tolerance:.6f} rad tolerance; "
                    "refusing to send a joint action."
                )

    @staticmethod
    def _quantize_joint_target(value: float) -> float:
        """Mirror pyAgxArm's millidegree wire encoding and return radians."""

        return math.radians(round(math.degrees(value) * 1000.0) / 1000.0)

    @staticmethod
    def _finite_action_value(key: str, value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Piper action {key!r} must be numeric.") from exc
        if not math.isfinite(result):
            raise ValueError(f"Piper action {key!r} must be finite.")
        return result


__all__ = [
    "PiperCameraTimeoutError",
    "PiperFeedbackError",
    "PiperRobot",
    "PiperRobotConfig",
]
