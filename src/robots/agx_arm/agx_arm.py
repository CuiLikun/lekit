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

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

try:
    import pyAgxArm as pyagxarm  # type: ignore[import-not-found]  # noqa: N813
except ImportError as e:
    raise ImportError("pyAgxArm is not installed") from e


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
    motors: list[str] = []

    def __init__(self, config: AgxArmConfig):
        super().__init__(config)
        self.config: AgxArmConfig = config
        self.arm: Any | None = None
        self.cameras: dict[str, Any] = make_cameras_from_configs(config.cameras) if config.cameras else {}

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """Schema for everything ``get_observation`` emits.

        The measured joints, gripper, flange pose, and configured camera frames
        are always emitted, independent of the active command representation.
        """
        features: dict[str, Any] = {f"{motor}.pos": float for motor in self.motors}
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
        features: dict[str, Any] = {f"{motor}.pos": float for motor in self.motors}
        features["gripper.pos"] = float
        features.update({f"ee.{axis}": float for axis in ("x", "y", "z", "roll", "pitch", "yaw")})
        return features

    @property
    def is_connected(self) -> bool:
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

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Open the CAN bus, configure the arm, and connect cameras."""
        sdk_config = pyagxarm.create_agx_arm_config(
            robot=self.config.arm_model,
            firmware_version=self.config.firmware_version,
            interface=self.config.interface,
            channel=self.config.channel,
            bitrate=self.config.bitrate,
        )
        self.arm = pyagxarm.AgxArmFactory.create_arm(sdk_config)

        try:
            self.arm.connect()
            if self.config.auto_clear_joint_error and self.arm.clear_joint_error() is False:
                print("AgxArm: controller did not acknowledge clear_joint_error().")
            if self.config.auto_enable:
                self.arm.enable()
            if self.config.auto_set_follower_mode:
                self.arm.set_follower_mode()

            self.effector = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            for camera in self.cameras.values():
                camera.connect()
            self.configure()
        except Exception:
            print("AgxArm: failed to connect; cleaning up.")
            self.disconnect()
            raise

        self.motors = [f"joint_{i + 1}" for i in range(self.arm.joint_nums)]

        print(f"AgxArm[{self.arm.joint_nums}DOF] connected.")
        print(f"firmware={self.arm.get_firmware()}")
        print(f"status={self.arm.is_ok()}")
        print(f"fps={self.arm.get_fps()}")
        print(f"arm_status={self.arm.get_arm_status()}")
        print(f"arm_model={self.config.arm_model}")
        print(f"id={self.id}")
        print(f"firmware_version={self.config.firmware_version}")
        print(f"interface={self.config.interface}")
        print(f"channel={self.config.channel}")
        print(f"joint_enable_status_list={self.arm.get_joints_enable_status_list()}")

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
                    print("AgxArm: could not disable joints during disconnect: %s", e)
            try:
                arm.disconnect()
            except Exception as e:
                print("AgxArm: error during disconnect: %s", e)
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception:
                print("AgxArm: camera disconnect failed, ignoring.")
        self.arm = None
        self.effector = None
        print("AgxArm[%s] disconnected.", self.id)

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        """Read arm state and configured camera frames."""
        obs: dict[str, Any] = {}
        obs.update(
            {f"{motor}.pos": angle for motor, angle in zip(self.motors, self.get_joint_angles(), strict=True)}
        )
        obs.update({"gripper.pos": self.get_gripper_status()})
        obs.update(
            {
                f"ee.{axis}": value
                for axis, value in zip(
                    ["x", "y", "z", "roll", "pitch", "yaw"], self.get_flange_pose(), strict=True
                )
            }
        )

        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                obs[camera_name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                obs[f"{camera_name}_depth"] = camera.read_latest_depth()
        return obs

    @check_if_not_connected
    def get_joint_angles(self) -> list[float]:
        """Read the configured model's measured joint angles in radians."""
        sample = self.arm.get_joint_angles()
        assert sample is not None
        angles = np.asarray(sample.msg, dtype=float)
        assert angles.shape == (self.arm.joint_nums,) and np.all(np.isfinite(angles))
        return [float(angle) for angle in angles]

    @check_if_not_connected
    def get_gripper_status(self) -> float:
        """Read normalized gripper position in [0, 1] as ``gripper.pos`` field."""
        status = None if self.effector is None else self.effector.get_gripper_status()
        assert status is not None
        assert status.msg.mode == "width"
        value = float(status.msg.value)
        assert np.isfinite(value)
        return value

    @check_if_not_connected
    def get_flange_pose(self) -> list[float]:
        sample = self.arm.get_flange_pose()
        assert sample is not None, "AgxArm: get_flange_pose() returned None."
        pose = np.asarray(sample.msg, dtype=float)
        assert pose.shape == (6,) and np.all(np.isfinite(pose))
        return [float(value) for value in pose]

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the configured joint-space or Cartesian control command."""
        if self.config.control_mode == "ee_pose":
            applied = self._send_ee_pose_action(action)
            applied.update(self.get_joint_angles())
        else:
            applied = self._send_joint_action(action)
            applied.update(self.get_flange_pose())
        return {key: float(applied[key]) for key in self.action_features}

    def _apply_gripper_action(self, action: dict[str, Any]) -> float:
        if "gripper.pos" not in action:
            return self.get_gripper_status()

        position = float(np.clip(self._as_finite_float(action["gripper.pos"], "gripper.pos"), 0.0, 1.0))
        if self.config.control_gripper and self.effector is not None:
            self.move_gripper_m(position * self.config.gripper_max_range)
        return position

    def _send_ee_pose_action(self, action: dict[str, Any]) -> dict[str, Any]:
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

    def _send_joint_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Clamp ``joint_N.pos`` to ``max_relative_target`` and call ``move_j``."""
        present_dict = self.get_joint_angles()
        present = [present_dict[f"{motor}.pos"] for motor in self.motors]
        goals: dict[str, float] = {}
        for motor in self.motors:
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
            command = [safe.get(f"{motor}.pos", present[index]) for index, motor in enumerate(self.motors)]
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
    from platform import system

    from rich import box
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    def format_vector(values: dict[str, Any]) -> str:
        """Render scalar robot values while omitting camera-frame arrays."""
        items = [
            f"{key}={float(value):.4f}"
            for key, value in values.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        ]
        return "[" + ", ".join(items) + "]"

    def make_live_panel(
        *,
        step: int,
        target_fps: float,
        actual_fps: float | None,
        observation: dict[str, Any],
        action: dict[str, Any],
        applied_action: dict[str, Any],
        arm_ok: bool,
    ) -> Panel:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold cyan")
        summary.add_column()
        summary.add_row("Step", str(step))
        summary.add_row("Loop FPS", "--" if actual_fps is None else f"{actual_fps:.2f}")
        summary.add_row("Target FPS", f"{target_fps:.2f}")
        summary.add_row("Arm OK", "[green]yes[/green]" if arm_ok else "[red]no[/red]")

        vectors = Table(box=box.SIMPLE_HEAVY, expand=True)
        vectors.add_column("Signal", style="bold cyan", no_wrap=True)
        vectors.add_column("Values", overflow="fold")
        vectors.add_row("State", format_vector(observation))
        vectors.add_row("Command", format_vector(action))
        vectors.add_row("Applied", format_vector(applied_action))

        return Panel(
            Group(summary, vectors),
            title="AGX Arm Live Monitor",
            border_style="cyan",
        )

    platform_system = system()
    if platform_system == "Windows":
        interface = "agx_cando"
        channel = "0"
    elif platform_system == "Linux":
        interface = "socketcan"
        channel = "can0"
    elif platform_system == "Darwin":
        interface = "slcan"
        channel = "/dev/ttyACM0"
    else:
        raise RuntimeError(
            "pyAgxArm currently documents Linux `socketcan`, Windows `agx_cando`, and macOS `slcan`."
        )

    cfg = AgxArmConfig(
        id="agx_demo",
        arm_model="piper_x",
        firmware_version="v188",
        interface=interface,
        channel=channel,
    )
    robot = AgxArm(cfg)
    fps = 1
    with robot:
        step = 0
        previous_frame_time = time.perf_counter()
        with Live(refresh_per_second=4, transient=False) as live:
            while True:
                frame_start = time.perf_counter()
                frame_interval = frame_start - previous_frame_time
                previous_frame_time = frame_start

                obs = robot.get_observation()
                pos = robot.get_flange_pose()
                new_pos = [p + 0.001 for p in pos]
                action: dict[str, Any] = {
                    "ee.x": new_pos[0],
                    "ee.y": new_pos[1],
                    "ee.z": new_pos[2],
                    "ee.roll": new_pos[3],
                    "ee.pitch": new_pos[4],
                    "ee.yaw": new_pos[5],
                    "gripper.pos": 0.5,
                }
                applied_action = robot.send_action(action)
                live.update(
                    make_live_panel(
                        step=step,
                        target_fps=fps,
                        actual_fps=1.0 / frame_interval if step else None,
                        observation=obs,
                        action=action,
                        applied_action=applied_action,
                        arm_ok=robot.arm.is_ok(),
                    )
                )
                step += 1
                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - frame_start)))
