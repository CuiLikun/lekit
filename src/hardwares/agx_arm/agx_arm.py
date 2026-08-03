"""LeRobot-compatible driver for AgileX robotic arms (Piper / Nero / ...).

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

* ``joint_1.pos`` … ``joint_6.pos`` — 6-DOF arm, observed and commanded in
  radians (``pyAgxArm`` is radians-native).
* ``gripper.pos`` — gripper width in **metres**, normalized by the Agx gripper
  range convention (``0.0`` = closed, ``gripper_max_range`` = fully open).
  This keeps the lerobot action schema independent of the Agx gripper's
  per-firmware range preset (``0.07`` m or ``0.1`` m).

Control mode: ``send_action`` issues a non-blocking ``move_j`` per arm cycle,
plus an ``effector.move_gripper_m`` for the gripper. ``move_j`` is
non-blocking (it returns once the joint command is on the bus), so successive
calls at the policy control rate don't stall.

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
    ...     robot.send_action({"joint_1.pos": 0.0, "joint_2.pos": 0.5, ..., "gripper.pos": 0.02})

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
    global _AGX, _AGX_FACTORY, _AGX_ARM_MODEL, _AGX_PIPER_FW, _AGX_NERO_FW
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
        _AGX_FACTORY = pyagxarm.AgxArmFactory
        _AGX_ARM_MODEL = pyagxarm.ArmModel
        _AGX_PIPER_FW = pyagxarm.PiperFW
        _AGX_NERO_FW = pyagxarm.NeroFW
    return _AGX


_AGX = None
_AGX_FACTORY = None
_AGX_ARM_MODEL = None
_AGX_PIPER_FW = None
_AGX_NERO_FW = None


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
    #   "joints"  — action is ``joint_1.pos..joint_6.pos + gripper.pos``; sent via
    #               ``move_j`` + ``move_gripper_m``. Use this when you want the host
    #               to do IK, or when the policy is trained in joint space.
    #   "ee_pose" — action is ``ee.{x,y,z,roll,pitch,yaw} + gripper.pos``; sent via
    #               ``move_p`` + ``move_gripper_m``. The AgileX firmware handles IK
    #               internally, so the XR / leader pipeline can skip the URDF-based
    #               IK step entirely.
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
    contract. ``get_observation`` returns the per-joint measured positions
    (radians) and the gripper width (metres) plus any configured camera
    frames. ``send_action`` issues a non-blocking ``move_j`` for the arm and
    a ``move_gripper_m`` for the gripper; successive calls at the policy
    control rate don't stall.

    Lifecycle: ``connect`` builds the SDK config + factory, opens the CAN
    bus, optionally enables joints / clears errors / engages follower mode,
    and connects any cameras. ``disconnect`` releases the firmware-side
    torque (when ``disable_torque_on_disconnect`` is True) and tears down
    the bus + cameras.
    """

    config_class: ClassVar[type] = AgxArmConfig
    name: ClassVar[str] = "agx_arm"

    # 6-DOF arm + gripper. Matches the pyAgxArm convention (``joint_1`` …
    # ``joint_6`` + a separately-initialized ``agx_gripper`` effector).
    N_JOINTS: ClassVar[int] = 6
    motors: ClassVar[list[str]] = [f"joint_{i + 1}" for i in range(N_JOINTS)] + ["gripper"]

    # ── Construction ──

    def __init__(self, config: AgxArmConfig):
        super().__init__(config)
        self.config: AgxArmConfig = config
        # Underlying pyAgxArm driver + effector, populated on ``connect``.
        self.arm: Any | None = None
        self.effector: Any | None = None
        # Cameras — built lazily so an ``AgxArmConfig`` with no cameras
        # doesn't pay the construction cost.
        self.cameras: dict[str, Any] = make_cameras_from_configs(config.cameras) if config.cameras else {}

    # ── Feature schemas ──

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """Schema for everything ``get_observation`` emits.

        Joints + gripper are always emitted (useful for diagnostics and
        for joint-mode policies). The current measured EE pose is emitted
        as ``ee.{x,y,z,roll,pitch,yaw}`` whenever ``control_mode == "ee_pose"``
        so that downstream consumers (e.g. a recording pipeline) can log
        the EE target without re-reading the SDK.
        """
        features: dict[str, Any] = {f"joint_{i + 1}.pos": float for i in range(self.N_JOINTS)}
        features["gripper.pos"] = float
        if self.config.control_mode == "ee_pose":
            for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
                features[f"ee.{axis}"] = float
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_key] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{cam_key}_depth"] = (cam.height, cam.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """Schema for the action dict ``send_action`` expects.

        ``control_mode == "joints"`` (default):
            ``joint_1.pos..joint_6.pos`` (radians) + ``gripper.pos`` (∈ [0, 1]).
            Dispatched to ``move_j`` + ``move_gripper_m``.

        ``control_mode == "ee_pose"``:
            ``ee.{x,y,z}`` (m, base frame) + ``ee.{roll,pitch,yaw}`` (radians,
            base frame) + ``gripper.pos`` (∈ [0, 1]). Dispatched to
            ``move_p`` + ``move_gripper_m``.
        """
        if self.config.control_mode == "ee_pose":
            return {f"ee.{axis}": float for axis in ("x", "y", "z", "roll", "pitch", "yaw")} | {
                "gripper.pos": float,
            }
        return {f"joint_{i + 1}.pos": float for i in range(self.N_JOINTS)} | {"gripper.pos": float}

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
        """Open the CAN bus, optionally enable joints, and connect cameras.

        Args:
            calibrate: Accepted for interface parity — ignored. The AgileX
                controller self-calibrates.
        """
        del calibrate  # Unused — see docstring.

        _require_pyagxarm()

        cfg = _AGX.create_agx_arm_config(
            robot=self.config.arm_model,
            firmeware_version=self.config.firmware_version,
            interface=self.config.interface,
            channel=self.config.channel,
            bitrate=self.config.bitrate,
        )
        arm = _AGX_FACTORY.create_arm(cfg)

        # Open the CAN bus + start the SDK's reader threads. ``connect`` is
        # idempotent in the SDK, so a re-entrant call here would no-op.
        arm.connect()
        self.arm = arm

        if self.config.auto_clear_joint_error:
            try:
                arm.clear_joint_error()
            except Exception:
                logger.debug("AgxArm: clear_joint_error unsupported / failed, ignoring.")

        if self.config.auto_enable:
            try:
                arm.enable()
            except Exception:
                logger.debug("AgxArm: enable() returned False / unsupported, ignoring.")

        if self.config.auto_set_follower_mode:
            try:
                arm.set_follower_mode()
            except Exception:
                logger.debug("AgxArm: set_follower_mode unsupported, ignoring.")

        # Initialise + connect the effector (only agx_gripper is wired up).
        if self.config.effector == "agx_gripper":
            try:
                self.effector = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            except Exception as e:
                logger.warning("AgxArm: failed to initialise agx_gripper effector: %s", e)
                self.effector = None

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(
            "AgxArm[%s %s fw=%s on %s/%s] connected.",
            self.config.arm_model,
            self.id,
            self.config.firmware_version,
            self.config.interface,
            self.config.channel,
        )

    def configure(self) -> None:
        """Apply runtime configuration. Called once at the end of ``connect``.

        Override-friendly — subclasses can extend this to install joint
        limits / payload settings / TCP offsets.
        """
        if not self.is_connected:
            return
        # ``move_j`` clamps joint targets when joint-limits are enabled and
        # the user has supplied overrides via the SDK config. We leave it
        # off here — the lerobot ``max_relative_target`` clamp is what we
        # actually enforce — but expose the hook so subclasses can flip it.
        if self.arm is not None and hasattr(self.arm, "set_joint_limits_enabled"):
            try:
                self.arm.set_joint_limits_enabled(False)
            except Exception:
                logger.debug("AgxArm: set_joint_limits_enabled failed, ignoring.")

    def disconnect(self) -> None:
        """Best-effort teardown: disable joints, drop the bus + cameras.

        Idempotent and safe to call when ``connect()`` never fully
        succeeded (e.g. CAN bus missing) -- the base ``Robot.__exit__``
        unconditionally calls ``disconnect()`` on context exit, so we
        can't let the ``@check_if_not_connected`` guard abort cleanup
        with a ``DeviceNotConnectedError`` that masks the original
        connection error.
        """
        arm = self.arm
        try:
            if arm is not None and self.config.disable_torque_on_disconnect:
                try:
                    #! WARNING: make sure the arm is safely parked before calling disable()
                    #! -- the arm will drop torque immediately and may fall if not supported.
                    # arm.disable()
                    ...
                except Exception:
                    logger.debug("AgxArm: disable() during disconnect failed, ignoring.")
            if arm is not None:
                arm.disconnect()
        except Exception as e:
            logger.warning("AgxArm: error during disconnect: %s", e)
        finally:
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception:
                    logger.debug("AgxArm: camera disconnect failed, ignoring.")
            self.arm = None
            self.effector = None
            logger.info("AgxArm[%s] disconnected.", self.id)

    # ── Observation / Action ──

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read joint angles (rad), gripper width (m), and any camera frames.

        When ``control_mode == "ee_pose"`` the current measured flange pose
        is included as ``ee.{x,y,z,roll,pitch,yaw}`` so downstream consumers
        can log the EE state without re-querying the SDK.
        """
        obs: dict[str, Any] = {}
        obs.update(self.get_joint_angles())
        obs.update(self.get_gripper_position())
        if self.config.control_mode == "ee_pose":
            obs.update(self.get_ee_pose())

        # Cameras.
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                obs[cam_key] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_key}_depth"] = cam.read_latest_depth()

        return obs

    @check_if_not_connected
    def get_joint_angles(self) -> dict[str, float]:
        """Read current 6-DOF joint angles in radians as ``joint_i.pos`` fields."""
        arm = self.arm
        assert arm is not None  # guaranteed by ``check_if_not_connected``

        # The SDK can return None before the first joint-state frame arrives.
        ja = arm.get_joint_angles()
        angles = [0.0] * self.N_JOINTS if ja is None else [float(v) for v in ja.msg]
        return {f"joint_{i + 1}.pos": float(angles[i]) for i in range(self.N_JOINTS)}

    @check_if_not_connected
    def get_gripper_position(self) -> dict[str, float]:
        """Read normalized gripper position in [0, 1] as ``gripper.pos`` field."""
        # The Agx gripper reports width in metres; normalize by configured range.
        value = 0.0
        if self.effector is not None and self.effector.get_gripper_status() is not None:
            gs = self.effector.get_gripper_status()
            value = float(gs.msg.value) if gs is not None else 0.0
            value = np.clip(value / self.config.gripper_max_range, 0.0, 1.0)

        return {"gripper.pos": float(value)}

    @check_if_not_connected
    def get_flange_pose(self) -> list[float]:
        fp = self.arm.get_flange_pose()
        assert fp is not None, "AgxArm.get_flange_pose: SDK returned None"
        return [float(v) for v in fp.msg]

    @check_if_not_connected
    def get_ee_pose(self) -> dict[str, float]:
        """Read the current flange pose as ``ee.{x,y,z,roll,pitch,yaw}`` fields.

        Used as the observation when ``control_mode == "ee_pose"`` so the
        recorded dataset captures the EE state (matching the EE-pose action
        schema).
        """
        pose = self.get_flange_pose()
        return {
            "ee.x": float(pose[0]),
            "ee.y": float(pose[1]),
            "ee.z": float(pose[2]),
            "ee.roll": float(pose[3]),
            "ee.pitch": float(pose[4]),
            "ee.yaw": float(pose[5]),
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Dispatch ``action`` to ``move_j`` or ``move_p`` based on its keys.

        Dispatch rules:

        * ``ee.x`` in ``action`` (and ``control_mode == "ee_pose"``) → EE
          pose path. ``ee.{x,y,z}`` (m, base frame) + ``ee.{roll,pitch,yaw}``
          (radians) are forwarded to ``move_p``; ``gripper.pos`` ∈ [0, 1]
          is scaled to ``gripper_max_range`` metres and sent via
          ``move_gripper_m``.
        * ``joint_N.pos`` in ``action`` → joint path. Per-joint targets are
          clamped against ``config.max_relative_target`` (radians) and
          issued via a non-blocking ``move_j``.
        * Otherwise, the controller's measured state is echoed back and
          nothing is commanded — useful for the idle frames of a record
          loop.

        Returns the (clamped) action that was actually sent to the
        controller so the caller can log or compare it against the raw
        policy output.
        """
        arm = self.arm
        assert arm is not None  # guaranteed by ``check_if_not_connected``

        if self.config.control_mode == "ee_pose" and "ee.x" in action:
            return self._send_ee_pose_action(action)
        if "joint_1.pos" in action:
            return self._send_joint_action(action)

        # No recognised control keys → idle frame, echo joints.
        applied = self.get_joint_angles()
        applied.update(self.get_gripper_position())
        if self.config.control_gripper and self.effector is not None and "gripper.pos" in action:
            try:
                width_m = (
                    float(np.clip(float(action["gripper.pos"]), 0.0, 1.0)) * self.config.gripper_max_range
                )
                self.effector.move_gripper_m(value=width_m, force=self.config.gripper_force)
                applied["gripper.pos"] = float(action["gripper.pos"])
            except Exception as e:
                logger.warning("AgxArm.send_action: idle-frame gripper command failed: %s", e)
        return applied

    # ── Internal send-action helpers ──

    def _send_ee_pose_action(self, action: RobotAction) -> RobotAction:
        """Forward ``ee.{x,y,z,roll,pitch,yaw} + gripper.pos`` to ``move_p`` + gripper."""
        arm = self.arm
        assert arm is not None

        try:
            pose = (
                float(action["ee.x"]),
                float(action["ee.y"]),
                float(action["ee.z"]),
                float(action["ee.roll"]),
                float(action["ee.pitch"]),
                float(action["ee.yaw"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise TypeError(f"AgxArm._send_ee_pose_action: malformed ee pose action: {e}") from e

        # Defence-in-depth per-frame rate limit. The XR pipeline already
        # rate-limits via EEBoundsAndSafety; this protects callers that
        # bypass that step (e.g. direct ``send_action`` from a policy).
        present_pose = self.get_flange_pose()
        dpos = np.array(pose[:3]) - np.array(present_pose[:3])
        step_m = float(np.linalg.norm(dpos))
        max_step = self.config.ee_max_step_m
        if max_step > 0.0 and step_m > max_step:
            scaled = np.array(present_pose[:3]) + dpos * (max_step / step_m)
            pose = (float(scaled[0]), float(scaled[1]), float(scaled[2]), pose[3], pose[4], pose[5])
            logger.warning(
                "AgxArm: EE jump %.3fm > %.3fm; rate-limited to per-frame step.",
                step_m,
                max_step,
            )

        try:
            arm.move_p(list(pose))
        except Exception as e:
            logger.warning("AgxArm._send_ee_pose_action: move_p failed: %s", e)

        applied: dict[str, float] = {
            f"ee.{axis}": float(pose[i]) for i, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw"))
        }

        if self.config.control_gripper and self.effector is not None and "gripper.pos" in action:
            try:
                width_m = (
                    float(np.clip(float(action["gripper.pos"]), 0.0, 1.0)) * self.config.gripper_max_range
                )
                self.effector.move_gripper_m(value=width_m, force=self.config.gripper_force)
                applied["gripper.pos"] = float(action["gripper.pos"])
            except Exception as e:
                logger.warning("AgxArm._send_ee_pose_action: gripper command failed: %s", e)
        return applied

    def _send_joint_action(self, action: RobotAction) -> RobotAction:
        """Clamp ``joint_N.pos`` to ``max_relative_target`` and call ``move_j``."""
        arm = self.arm
        assert arm is not None

        present_dict = self.get_joint_angles()
        present = [present_dict[f"joint_{i + 1}.pos"] for i in range(self.N_JOINTS)]

        # Filter down to the joints the caller is actually driving. Unknown
        # keys are ignored so the caller can pass arbitrary telemetry fields
        # (e.g. status flags) without breaking the contract.
        goals: dict[str, float] = {}
        for i in range(self.N_JOINTS):
            key = f"joint_{i + 1}.pos"
            if key in action:
                try:
                    goals[key] = float(action[key])
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"AgxArm.send_action: action[{key!r}]={action[key]!r} is not numeric"
                    ) from e

        safe: dict[str, float] = {}
        if goals and self.config.max_relative_target is not None:
            safe = ensure_safe_goal_position(
                {k: (goals[k], present_dict[k]) for k in goals},
                self.config.max_relative_target,
            )
        elif goals:
            safe = dict(goals)

        # Build the 6-tuple move_j target in joint order. Joints the caller
        # didn't touch fall back to their present position so the SDK never
        # sees ``None`` entries.
        cmd = tuple(safe.get(f"joint_{i + 1}.pos", present[i]) for i in range(self.N_JOINTS))
        try:
            arm.move_j(list(cmd))
        except Exception as e:
            logger.warning("AgxArm.send_action: move_j failed: %s", e)

        applied_action: dict[str, float] = dict(present_dict)
        applied_action.update(safe)
        if self.config.control_gripper and self.effector is not None and "gripper.pos" in action:
            try:
                width_m = (
                    float(np.clip(float(action["gripper.pos"]), 0.0, 1.0)) * self.config.gripper_max_range
                )
                self.effector.move_gripper_m(value=width_m, force=self.config.gripper_force)
                applied_action["gripper.pos"] = float(action["gripper.pos"])
            except Exception as e:
                logger.warning("AgxArm.send_action: gripper command failed: %s", e)
        return applied_action

    # ── Direct motion commands (bypass the per-step safety clamp) ──

    @check_if_not_connected
    def move_j(self, joints: list[float] | tuple[float, ...]) -> None:
        """Joint-space ``move_j`` (radians). Bypasses ``max_relative_target``."""
        if self.arm is None:
            raise RuntimeError("AgxArm: arm is not connected.")
        self.arm.move_j(list(joints))

    def move_p(self, pose: list[float] | tuple[float, ...]) -> None:
        """Cartesian-space ``move_p`` (x, y, z in metres; roll, pitch, yaw in radians).

        Bypasses ``max_relative_target``.
        """
        if self.arm is None:
            raise RuntimeError("AgxArm: arm is not connected.")
        self.arm.move_p(list(pose))

    @check_if_not_connected
    def move_gripper_m(self, value: float, force: float | None = None) -> None:
        """Drive the gripper to ``value`` metres (closedness = 0 at ``0``)."""
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
