"""LeRobot-compatible driver for AgileX robotic arms (Piper / Nero).

This module wraps the :mod:`pyAgxArm` Python SDK to implement
:class:`AgxArm`, a drop-in :class:`lerobot.robots.robot.Robot` subclass for
the AgileX CAN-bus arms currently sold under ``piper``, ``piper_h``,
``piper_l``, ``piper_x`` and ``nero`` model names.

The driver is hardware-agnostic w.r.t. the exact model: the user picks the
model + firmware via ``AgxArmConfig``, and the matching ``pyAgxArm`` driver
class is resolved through :func:`pyAgxArm.AgxArmFactory.create_arm`. Only the
``agx_gripper`` effector is wired up; ``revo2`` / ``revo2_touch`` effectors
are recognized by the SDK but not surfaced here.

Joint schema (matches the SDK convention):

* ``joint_1.pos`` … ``joint_N.pos`` — 6 DOF for Piper variants and 7 DOF for
  Nero, observed and commanded in radians (``pyAgxArm`` is radians-native).
* ``gripper.pos`` — normalized gripper opening (``0.0`` = closed, ``1.0`` =
  fully open). SDK feedback and commands are converted to metres using
  ``gripper_max_range`` (``0.07`` m or ``0.1`` m).

Control mode: ``send_action`` issues ``move_j`` or ``move_p`` according to the
configured mode, plus an ``effector.move_gripper_m`` for the gripper.

Example:

    >>> from hardwares.agx_arm import AgxArm, AgxArmConfig
    >>> cfg = AgxArmConfig(
    ...     arm_model="piper_x",
    ...     firmware_version="v188",
    ...     channel="can0",
    ...     interface="socketcan",
    ... )
    >>> robot = AgxArm(cfg)
    >>> with robot:
    ...     obs = robot.get_observation()
    ...     robot.send_action({"joint_1.pos": 0.0, "joint_2.pos": 0.5, ..., "gripper.pos": 0.5})

See https://github.com/agilexrobotics/pyAgxArm for the upstream SDK.
"""

from __future__ import annotations

import logging
import time
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

logger = logging.getLogger(__name__)


# ── Lazy pyAgxArm import ────────────────────────────────────────────────────
#
# ``pyAgxArm`` requires a running CAN stack (Linux socketcan / Windows
# ``agx_cando`` / macOS ``slcan``) and pulls in the ``can`` package. The
# factory + driver imports happen lazily so the module can be imported in
# environments where ``pyAgxArm`` is unavailable (CI, mock_robot tests,
# ``lerobot.robots.utils`` factory probing).


def _require_pyagxarm():
    """Import pyAgxArm lazily and cache the module-level symbols we use."""
    global _AGX
    if _AGX is None:
        try:
            import pyAgxArm as pyagxarm  # type: ignore[import-not-found]  # noqa: N813
        except ImportError as e:
            raise ImportError(
                "pyAgxArm is required to use AgxArm. Install it with "
                "`pip install pyAgxArm` (or the upstream repo "
                "`agilexrobotics/pyAgxArm`)."
            ) from e
        _AGX = pyagxarm
    return _AGX


_AGX = None


# ── Config ──────────────────────────────────────────────────────────────────


# Model -> firmware-constant lookup. The factory accepts string literals
# (``"piper_x"``, ``"v188"``), so the mapping below is a convenience for
# callers that pass :class:`AgxArmConfig` kwargs.
_PIPER_MODELS = {"piper", "piper_h", "piper_l", "piper_x"}
_NERO_MODELS = {"nero"}
_PIPER_FIRMWARES = {"default", "v183", "v188", "v189"}
_NERO_FIRMWARES = {"default", "v111", "v112", "v120"}


def _resolve_arm_model(model: str) -> str:
    if model in _PIPER_MODELS or model in _NERO_MODELS:
        return model
    raise ValueError(
        f"AgxArmConfig.arm_model={model!r} is not a known AgileX model. "
        f"Expected one of: {sorted(_PIPER_MODELS | _NERO_MODELS)}."
    )


def _resolve_firmware(model: str, firmware: str) -> str:
    valid: set[str] | None = None
    family = ""
    if model in _PIPER_MODELS:
        valid = _PIPER_FIRMWARES
        family = "Piper"
    elif model in _NERO_MODELS:
        valid = _NERO_FIRMWARES
        family = "Nero"
    if valid is not None and firmware not in valid:
        raise ValueError(
            f"AgxArmConfig.firmware_version={firmware!r} is not a known {family} "
            f"firmware. Expected one of: {sorted(valid)}."
        )
    return firmware


@RobotConfig.register_subclass("agx_arm")
@dataclass
class AgxArmConfig(RobotConfig):
    """Configuration for :class:`AgxArm`.

    All fields are keyword-only (inherited from ``RobotConfig``). When loaded
    from YAML, ``type: agx_arm`` selects this configuration.
    """

    # AgileX arm model identifier (e.g. ``"piper_x"``, ``"piper"``, ``"nero"``).
    arm_model: str = "piper_x"
    # Firmware version string (e.g. ``"v188"`` for the Piper X in the demo).
    firmware_version: str = "v188"

    # CAN interface / channel. ``socketcan`` + ``can0`` matches a Linux
    # SocketCAN setup; Windows / macOS users typically swap to ``agx_cando``
    # / ``slcan`` + the matching USB channel.
    interface: str = "socketcan"
    channel: str = "can0"
    bitrate: int = 1_000_000

    # Effector kind (only ``agx_gripper`` is wired up here).
    effector: str = "agx_gripper"

    # Whether ``connect()`` also powers-on and enables the servos so the
    # first ``send_action`` is immediately ready.
    auto_enable: bool = True
    # Best-effort clear of any joint-error state on connect (matches
    # demo.py: ``robot.clear_joint_error()``).
    auto_clear_joint_error: bool = True
    # Engage the firmware's leader/follower linkage mode (follower side).
    # Demo.py calls ``robot.set_follower_mode()`` before driving commands;
    # we keep the same default.
    auto_set_follower_mode: bool = True

    # Disable joints on disconnect (matches the lerobot convention).
    disable_torque_on_disconnect: bool = True

    # Per-joint ``move_j`` relative-position safety clamp (radians). ``None``
    # disables the clamp. Defaults to ~2.86° which is conservative for a
    # 30 Hz control loop on a Piper X. Only applies in ``joints`` mode; in
    # ``ee_pose`` mode the EE pose is rate-limited by ``ee_max_step_m``.
    max_relative_target: float | dict[str, float] | None = 0.05

    # Action schema / control backend:
    #   "joints"  — joint targets are sent via ``move_j``; Cartesian fields are
    #               returned from feedback.
    #   "ee_pose" — Cartesian targets are sent via ``move_p``; joint fields are
    #               returned from feedback. The AgileX firmware handles IK
    #               internally, so the XR pipeline can skip host-side IK.
    # Type is ``str`` (not ``Literal[...]``) because draccus's CLI decoder
    # does not understand ``typing.Literal``; ``__post_init__`` validates.
    control_mode: str = "joints"

    # Per-frame Cartesian step clamp [m]. Only meaningful in ``ee_pose`` mode;
    # the lerobot ``EEBoundsAndSafety`` step in the XR pipeline applies its
    # own per-frame limit, so this is a defence-in-depth clamp that survives
    # even when the caller bypasses that step.
    ee_max_step_m: float = 0.1

    # Agx gripper configuration. ``max_range`` is the firmware-reported
    # full-open width in metres (0.07 m or 0.1 m depending on the gripper
    # SKU). The lerobot ``gripper.pos`` action is normalized to [0, 1] and
    # scaled to metres internally.
    gripper_max_range: float = 0.07
    # Gripper force [N] passed to ``move_gripper_m`` on every command.
    gripper_force: float = 10.0
    # Whether ``send_action`` should drive the gripper. When False, the
    # gripper observation is still emitted but the gripper is left under
    # firmware default behaviour.
    control_gripper: bool = True

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Normalize and validate model / firmware eagerly so misconfigurations
        # surface at parse time, not at first ``connect()``.
        self.arm_model = _resolve_arm_model(self.arm_model)
        self.firmware_version = _resolve_firmware(self.arm_model, self.firmware_version)
        if self.effector != "agx_gripper":
            raise ValueError(
                f"AgxArmConfig.effector={self.effector!r} is not supported. "
                "Only 'agx_gripper' is wired up by AgxArm."
            )
        if self.bitrate <= 0:
            raise ValueError("AgxArmConfig.bitrate must be positive.")
        if self.gripper_max_range <= 0:
            raise ValueError("AgxArmConfig.gripper_max_range must be positive.")
        if self.gripper_force <= 0:
            raise ValueError("AgxArmConfig.gripper_force must be positive.")
        # 0.07 / 0.1 m are the only presets the Agx gripper firmware
        # actually accepts (see ``set_gripper_teaching_pendant_param``).
        if self.gripper_max_range not in (0.07, 0.1):
            raise ValueError(
                f"AgxArmConfig.gripper_max_range={self.gripper_max_range} is not one of "
                "the Agx gripper's accepted presets (0.07 or 0.1 m)."
            )
        if self.max_relative_target is not None and not (
            (isinstance(self.max_relative_target, (int, float)) and self.max_relative_target > 0)
            or isinstance(self.max_relative_target, dict)
        ):
            raise ValueError(
                "AgxArmConfig.max_relative_target must be a positive scalar or a "
                "dict of motor_name -> positive scalar, or None to disable."
            )
        if self.control_mode not in ("joints", "ee_pose"):
            raise ValueError(
                f"AgxArmConfig.control_mode={self.control_mode!r} is not supported. "
                "Expected 'joints' or 'ee_pose'."
            )
        if self.ee_max_step_m <= 0:
            raise ValueError("AgxArmConfig.ee_max_step_m must be positive.")


# ── Robot implementation ────────────────────────────────────────────────────


class AgxArm(Robot):
    """LeRobot-compatible AgileX arm driver.

    The class follows the standard :class:`lerobot.robots.robot.Robot`
    contract. Piper variants expose six joints and Nero exposes seven. The
    SDK receives Cartesian targets directly in ``ee_pose`` mode, so no host
    inverse-kinematics solver is required.

    Lifecycle: ``connect`` builds the SDK config + factory, opens the CAN
    bus, optionally enables joints / clears errors / engages follower mode,
    and connects any cameras. ``disconnect`` releases the firmware-side
    torque (when ``disable_torque_on_disconnect`` is True) and tears down
    the bus + cameras.
    """

    config_class: ClassVar[type] = AgxArmConfig
    name: ClassVar[str] = "agx_arm"

    # Piper variants have six joints; Nero has seven. The instance value is
    # selected from the model so its schema matches the SDK command length.
    N_JOINTS: ClassVar[int] = 6
    motors: ClassVar[list[str]] = [f"joint_{i + 1}" for i in range(N_JOINTS)] + ["gripper"]
    _JOINT_COUNTS: ClassVar[dict[str, int]] = {"nero": 7, **dict.fromkeys(_PIPER_MODELS, 6)}

    # ── Construction ──

    def __init__(self, config: AgxArmConfig):
        super().__init__(config)
        self.config: AgxArmConfig = config
        self.N_JOINTS = self._JOINT_COUNTS[config.arm_model]
        self.arm_motors = [f"joint_{index + 1}" for index in range(self.N_JOINTS)]
        self.motors = [*self.arm_motors, "gripper"]
        self.arm: Any | None = None
        self.effector: Any | None = None
        self.cameras: dict[str, Any] = make_cameras_from_configs(config.cameras) if config.cameras else {}

    # ── Feature schemas ──

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """Schema for everything ``get_observation`` emits.

        The measured joints, gripper, flange pose, and configured camera frames
        are always emitted, independent of the active command representation.
        """
        features: dict[str, Any] = {f"{motor}.pos": float for motor in self.arm_motors}
        features["gripper.pos"] = float
        features.update({f"ee.{axis}": float for axis in ("x", "y", "z", "roll", "pitch", "yaw")})
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_key] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{cam_key}_depth"] = (cam.height, cam.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """Schema for the action dict ``send_action`` expects.

        The schema always contains both ``joint_N.pos`` and
        ``ee.{x,y,z,roll,pitch,yaw}`` fields, plus ``gripper.pos``. The active
        ``control_mode`` selects which arm representation is sent to the SDK.
        """
        features: dict[str, Any] = {f"{motor}.pos": float for motor in self.arm_motors}
        features.update({f"ee.{axis}": float for axis in ("x", "y", "z", "roll", "pitch", "yaw")})
        features["gripper.pos"] = float
        return features

    # ── Connection state ──

    @property
    def is_connected(self) -> bool:
        # Match the lerobot convention (see SOFollower / OpenArmFollower):
        # the connected check covers the arm bus + cameras only. The
        # effector's ``is_ok()`` is a runtime health signal (CAN traffic
        # flowing, no driver fault) that can flap independently of the
        # bus state, so it is exposed separately rather than gating
        # ``get_observation`` / ``send_action``.
        return (
            self.arm is not None
            and self.arm.is_connected()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # The AgileX controller handles joint zeroing internally; the host
        # has no per-motor calibration step (matches the demo workflow).
        return True

    def calibrate(self) -> None:
        """No-op: the AgileX controller self-calibrates; nothing to do here."""
        return

    # ── Lifecycle ──

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Open the CAN bus, configure the arm, and connect cameras."""
        del calibrate

        sdk = _require_pyagxarm()
        sdk_config = sdk.create_agx_arm_config(
            robot=self.config.arm_model,
            firmeware_version=self.config.firmware_version,
            interface=self.config.interface,
            channel=self.config.channel,
            bitrate=self.config.bitrate,
        )
        arm = sdk.AgxArmFactory.create_arm(sdk_config)
        self.arm = arm

        try:
            arm.connect()
            if self.config.auto_clear_joint_error and arm.clear_joint_error() is False:
                logger.warning("AgxArm: controller did not acknowledge clear_joint_error().")
            if self.config.auto_enable:
                self._enable_arm(arm)
            if self.config.auto_set_follower_mode:
                arm.set_follower_mode()

            self.effector = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            for camera in self.cameras.values():
                camera.connect()
            self.configure()
        except Exception:
            self.disconnect()
            raise
        logger.info(
            "AgxArm[%s %s fw=%s on %s/%s] connected.",
            self.config.arm_model,
            self.id,
            self.config.firmware_version,
            self.config.interface,
            self.config.channel,
        )

    @staticmethod
    def _enable_arm(arm: Any) -> None:
        deadline = time.monotonic() + 2.0
        while not arm.enable():
            if time.monotonic() >= deadline:
                raise RuntimeError("AgxArm: timed out waiting for all joints to enable.")
            time.sleep(0.01)

    def configure(self) -> None:
        """Enable the SDK's model-specific software joint limits."""
        if self.arm is not None:
            self.arm.set_joint_limits_enabled(True)

    def disconnect(self) -> None:
        """Best-effort teardown of torque, CAN resources, and cameras."""
        arm = self.arm
        if arm is not None:
            if self.config.disable_torque_on_disconnect:
                try:
                    if arm.is_connected():
                        arm.disable()
                except Exception as e:
                    logger.warning("AgxArm: could not disable joints during disconnect: %s", e)
            try:
                arm.disconnect()
            except Exception as e:
                logger.warning("AgxArm: error during disconnect: %s", e)
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception:
                logger.debug("AgxArm: camera disconnect failed, ignoring.")
        self.arm = None
        self.effector = None
        logger.info("AgxArm[%s] disconnected.", self.id)

    # ── Observation / Action ──

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read arm state and configured camera frames."""
        obs: dict[str, Any] = {}
        obs.update(self.get_joint_angles())
        obs.update(self.get_gripper_position())
        obs.update(self.get_ee_pose())

        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                obs[camera_name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                obs[f"{camera_name}_depth"] = camera.read_latest_depth()
        return obs

    @check_if_not_connected
    def get_joint_angles(self) -> dict[str, float]:
        """Read the configured model's measured joint angles in radians."""
        sample = self.arm.get_joint_angles()
        if sample is None:
            return dict.fromkeys((f"{motor}.pos" for motor in self.arm_motors), 0.0)

        angles = np.asarray(sample.msg, dtype=float)
        if angles.shape != (self.N_JOINTS,) or not np.all(np.isfinite(angles)):
            raise RuntimeError(f"AgxArm: invalid joint feedback {sample.msg!r}.")
        return {f"{motor}.pos": float(angles[index]) for index, motor in enumerate(self.arm_motors)}

    @check_if_not_connected
    def get_gripper_position(self) -> dict[str, float]:
        """Read normalized gripper position in [0, 1] as ``gripper.pos`` field."""
        status = None if self.effector is None else self.effector.get_gripper_status()
        if status is None:
            return {"gripper.pos": 0.0}
        if status.msg.mode != "width":
            raise RuntimeError("AgxArm: AgxGripper must be in width mode to report gripper.pos.")

        value = float(status.msg.value)
        if not np.isfinite(value):
            raise RuntimeError(f"AgxArm: invalid gripper feedback {value!r}.")
        return {"gripper.pos": float(np.clip(value / self.config.gripper_max_range, 0.0, 1.0))}

    @check_if_not_connected
    def get_flange_pose(self) -> list[float]:
        sample = self.arm.get_flange_pose()
        if sample is None:
            raise RuntimeError("AgxArm: no flange-pose feedback has been received.")

        pose = np.asarray(sample.msg, dtype=float)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"AgxArm: invalid flange-pose feedback {sample.msg!r}.")
        return [float(value) for value in pose]

    @check_if_not_connected
    def get_ee_pose(self) -> dict[str, float]:
        """Read the current flange pose as Cartesian action fields."""
        return dict(
            zip(
                ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw"),
                self.get_flange_pose(),
                strict=True,
            )
        )

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Dispatch the configured joint-space or Cartesian control command."""
        if self.config.control_mode == "ee_pose":
            applied = self._send_ee_pose_action(action)
            applied.update(self.get_joint_angles())
        else:
            applied = self._send_joint_action(action)
            applied.update(self.get_ee_pose())
        return {key: float(applied[key]) for key in self.action_features}

    @staticmethod
    def _as_finite_float(value: Any, key: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"AgxArm.send_action: action[{key!r}]={value!r} is not numeric") from e
        if not np.isfinite(result):
            raise ValueError(f"AgxArm.send_action: action[{key!r}] must be finite")
        return result

    def _apply_gripper_action(self, action: RobotAction) -> float:
        if "gripper.pos" not in action:
            return self.get_gripper_position()["gripper.pos"]

        position = float(np.clip(self._as_finite_float(action["gripper.pos"], "gripper.pos"), 0.0, 1.0))
        if self.config.control_gripper and self.effector is not None:
            self.move_gripper_m(position * self.config.gripper_max_range)
        return position

    def _send_ee_pose_action(self, action: RobotAction) -> RobotAction:
        """Rate-limit a flange target before passing it to SDK Cartesian control."""
        keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
        current = np.asarray(self.get_flange_pose(), dtype=float)
        target = current.copy()
        for index, key in enumerate(keys):
            if key in action:
                target[index] = self._as_finite_float(action[key], key)

        delta = target[:3] - current[:3]
        distance = float(np.linalg.norm(delta))
        if distance > self.config.ee_max_step_m:
            target[:3] = current[:3] + delta * (self.config.ee_max_step_m / distance)

        self.move_p(target)
        applied = {key: float(target[index]) for index, key in enumerate(keys)}
        applied["gripper.pos"] = self._apply_gripper_action(action)
        return applied

    def _send_joint_action(self, action: RobotAction) -> RobotAction:
        """Clamp ``joint_N.pos`` to ``max_relative_target`` and call ``move_j``."""
        present_dict = self.get_joint_angles()
        present = [present_dict[f"{motor}.pos"] for motor in self.arm_motors]
        goals: dict[str, float] = {}
        for motor in self.arm_motors:
            key = f"{motor}.pos"
            if key in action:
                goals[key] = self._as_finite_float(action[key], key)

        if goals and self.config.max_relative_target is not None:
            safe = ensure_safe_goal_position(
                {k: (goals[k], present_dict[k]) for k in goals},
                self.config.max_relative_target,
            )
        elif goals:
            safe = dict(goals)
        else:
            safe = {}

        if goals:
            command = [
                safe.get(f"{motor}.pos", present[index]) for index, motor in enumerate(self.arm_motors)
            ]
            self.move_j(command)

        applied = {**present_dict, **safe}
        applied["gripper.pos"] = self._apply_gripper_action(action)
        return applied

    # ── Direct motion commands (bypass the per-step safety clamp) ──

    @check_if_not_connected
    def move_j(self, joints: list[float] | tuple[float, ...]) -> None:
        """Joint-space ``move_j`` (radians). Bypasses ``max_relative_target``."""
        if self.arm is None:
            raise RuntimeError("AgxArm: arm is not connected.")
        self.arm.move_j(list(joints))

    @check_if_not_connected
    def move_p(self, pose: list[float] | tuple[float, ...] | np.ndarray) -> None:
        """Cartesian-space ``move_p`` (x, y, z in metres; roll, pitch, yaw in radians).

        Bypasses ``max_relative_target``.
        """
        if self.arm is None:
            raise RuntimeError("AgxArm: arm is not connected.")
        self.arm.move_p(list(pose))

    @check_if_not_connected
    def move_gripper_m(self, value: float, force: float | None = None) -> None:
        """Drive the gripper to ``value`` metres of opening width."""
        if self.effector is None:
            raise RuntimeError("AgxArm: gripper effector is not initialized.")
        self.effector.move_gripper_m(
            value=float(value),
            force=self.config.gripper_force if force is None else float(force),
        )


__all__ = [
    "AgxArm",
    "AgxArmConfig",
]


if __name__ == "__main__":
    from rich import print

    logging.basicConfig(level=logging.INFO)

    cfg = AgxArmConfig(
        id="agx_demo",
        arm_model="piper_x",
        firmware_version="v188",
        interface="socketcan",
        channel="can0",
    )
    robot = AgxArm(cfg)
    print(robot)
    fps = 10.0
    with robot:
        step = 0
        while True:
            time.sleep(1.0 / fps)
            obs = robot.get_observation()
            pos = robot.get_flange_pose()
            new_pos = [p + 0.001 for p in pos]
            with np.printoptions(precision=4, suppress=True):
                print(obs)
            # robot.send_action(
            #     {
            #         "ee.x": new_pos[0],
            #         "ee.y": new_pos[1],
            #         "ee.z": new_pos[2],
            #         "ee.roll": new_pos[3],
            #         "ee.pitch": new_pos[4],
            #         "ee.yaw": new_pos[5],
            #         "gripper.pos": 0.5,
            #     }
            # )
            step += 1
