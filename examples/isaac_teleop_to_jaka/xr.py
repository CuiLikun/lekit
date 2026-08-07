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

"""Isaac Teleop XR controller integration for the JAKA recorder.

Bundles:
  * ``IsaacTeleopConfig`` / ``XRControllerConfig`` — dataclass configs registered with
    ``--teleop.type=xr_controller`` so draccus can decode the CLI.
  * ``XRController`` — thin reader of the raw base-frame grip pose off an Isaac Teleop
    ``ControllersSource``. No retargeting (the recorder rebases via :class:`Clutch`).
  * ``make_xr_device`` — wires the controller, clutch, and rotvec->RPY conversion into
    a JAKA ``action_features`` dict the recorder can hand to :meth:`JakaRobot.send_action`.
"""

from __future__ import annotations

import abc
import logging
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.import_utils import is_package_available
from lerobot.utils.rotation import Rotation
from robots.jaka_robot.pose_math import matrix_to_rpy

from .clutch import Clutch

logger = logging.getLogger(__name__)

# Fixed poll rate for the pre-loop waits. Same value the original common module used; only
# consumed by the connect-wait polling, not the control loop.
POLL_HZ = 30

# Max per-frame EE translation step [m/frame]. Guards the rotvec->RPY path against a
# tracking glitch producing a one-frame teleport. JAKA's own per-frame clamp is stricter
# (0.01 m); this is defence in depth.
MAX_EE_STEP_M = 0.05

# How often to re-print the CloudXR connection hint while waiting for the headset [s].
_CONNECT_REMINDER_S = 15.0

# CloudXR device-profile env file; required for the auto-launched CloudXR runtime.
CLOUDXR_ENV_FILE = str(files(__package__) / "default.env")

# Virtual / bridge / USB-gadget interfaces a headset can't reach over the network —
# listed by prefix so the connect hint skips them.
_SKIP_IFACE_PREFIXES = ("docker", "br-", "veth", "virbr", "l4tbr")

# CloudXR web client URL opened in the headset.
_CLOUDXR_WEB_CLIENT_URL = "https://nvidia.github.io/IsaacTeleop/client"
# WSS-proxy / self-signed-cert port the operator accepts in-browser before connecting.
_CLOUDXR_WSS_PORT = 48322

# isaacteleop is an optional NVIDIA dep. Guard the import so this module loads without it
# (and construction fails fast with install instructions instead of at first use).
_isaacteleop_available = is_package_available("isaacteleop")

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.cloudxr import CloudXRLauncher
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import (
        ExecutionEvents,
        ExecutionState,
        OutputCombiner,
        TensorGroup,
        ValueInput,
    )
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
    from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig
else:
    CloudXRLauncher = None
    ControllersSource = None
    ExecutionEvents = None
    ExecutionState = None
    OutputCombiner = None
    TensorGroup = None
    ValueInput = None
    TransformMatrix = None
    ControllerInputIndex = None
    TeleopSession = None
    TeleopSessionConfig = None

# Static rebase from the CloudXR controller anchor frame into the JAKA base frame used by
# this setup. The X/Y signs are selected for the physical arm mounting: lateral hand motion
# maps to lateral EE motion, and forward/backward hand motion maps to EE X. ``base_T_anchor``
# remains configurable for a differently mounted arm.
_DEFAULT_BASE_T_ANCHOR: list[list[float]] = [
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# Source-node name for the static base_T_anchor rebase input fed via TeleopSession.step.
_BASE_T_ANCHOR_INPUT = "base_T_anchor"


# ======================================================================================
# Configs
# ======================================================================================


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared base for Isaac Teleop-backed teleoperators.

    Uses its own draccus ``_choice_registry`` (decoupled from the global
    :class:`TeleoperatorConfig` one) so ``--teleop.type`` on a field typed
    ``IsaacTeleopConfig`` resolves against ONLY the Isaac devices — letting them claim
    short names (``xr_controller``) without colliding with the global registry.
    """

    _choice_registry: ClassVar[dict] = {}

    app_name: str = "LeTeleop"
    """Application name for the OpenXR / Isaac Teleop session."""

    auto_launch_cloudxr: bool = True
    """Auto-launch the CloudXR runtime on :meth:`connect`. Set ``False`` (or export
    ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1``) when CloudXR runs externally."""

    cloudxr_env_file: str | None = None
    """Optional CloudXR device-profile ``.env`` passed to ``CloudXRLauncher``."""


@IsaacTeleopConfig.register_subclass("xr_controller")
@dataclass(kw_only=True)
class XRControllerConfig(IsaacTeleopConfig):
    """Config for Isaac Teleop XR (VR) controller teleoperation.

    Exposes the raw base-frame grip pose, squeeze, and trigger; no retargeters.
    """

    hand_side: str = "right"
    """Which controller hand to use: ``"left"`` or ``"right"``."""

    clutch_threshold: float = 0.5
    """Squeeze value above which the recorder's clutch engages (held-to-enable)."""

    lock_pose: bool = False
    """Keep the measured JAKA roll/pitch/yaw fixed while the clutch is engaged."""

    base_T_anchor: list[list[float]] = field(  # noqa: N815
        default_factory=lambda: [row.copy() for row in _DEFAULT_BASE_T_ANCHOR]
    )
    """Static 4x4 transform rebasing the OpenXR anchor frame into the robot base frame."""

    def __post_init__(self):
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {self.hand_side!r}")
        if not isinstance(self.lock_pose, bool):
            raise ValueError("lock_pose must be a boolean")


# ======================================================================================
# Base teleoperator
# ======================================================================================


def _require_isaacteleop() -> None:
    if not _isaacteleop_available:
        raise ImportError(
            "The 'isaacteleop' package is required for Isaac Teleop devices but is not "
            "installed. See examples/isaac_teleop_to_jaka/README.md for install instructions."
        )


class IsaacTeleopTeleoperator(Teleoperator):
    """Base class for teleoperators backed by an Isaac Teleop ``TeleopSession``.

    Owns the session lifecycle, the per-step health guard, and the CloudXR runtime.
    Subclasses supply :meth:`_build_pipeline` and :meth:`get_action`.
    """

    config_class = IsaacTeleopConfig

    def __init__(self, config: IsaacTeleopConfig):
        _require_isaacteleop()
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self._session: TeleopSession | None = None
        self._cloudxr_launcher: CloudXRLauncher | None = None

    @abc.abstractmethod
    def _build_pipeline(self) -> OutputCombiner:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Tracking devices are self-calibrating.

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        self._ensure_cloudxr_runtime()

        try:
            pipeline = self._build_pipeline()
            session_config = TeleopSessionConfig(app_name=self.config.app_name, pipeline=pipeline)
            self._session = TeleopSession(session_config)
            self._session.__enter__()
        except Exception:
            self._session = None
            try:
                self._stop_cloudxr_runtime()
            except Exception:
                logger.exception("Failed to stop CloudXR runtime during connect() rollback")
            raise
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def disconnect(self) -> None:
        try:
            if self._session is not None:
                # Null the handle BEFORE __exit__: even a failed session teardown must not
                # wedge the device as is_connected.
                session = self._session
                self._session = None
                session.__exit__(None, None, None)
                logger.info("Isaac Teleop session ended")
        finally:
            self._stop_cloudxr_runtime()

    def _ensure_cloudxr_runtime(self) -> None:
        """Auto-launch the CloudXR runtime once, unless opted out."""
        if self._cloudxr_launcher is not None:
            return

        if os.environ.get("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "").strip() == "1":
            logger.info("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1 set; skipping CloudXR auto-launch.")
            return

        if not self.config.auto_launch_cloudxr:
            logger.info("config.auto_launch_cloudxr is False; skipping CloudXR auto-launch.")
            return

        logger.info("Launching CloudXR runtime (first run may prompt for EULA and take ~30s)...")
        self._cloudxr_launcher = CloudXRLauncher(
            install_dir=str(Path.home() / ".cloudxr"),
            env_config=self.config.cloudxr_env_file,
            accept_eula=False,
        )

    def _stop_cloudxr_runtime(self) -> None:
        if self._cloudxr_launcher is None:
            return
        try:
            self._cloudxr_launcher.stop()
        except RuntimeError:
            logger.warning("CloudXR runtime could not be terminated; handle retained for atexit cleanup")
        else:
            self._cloudxr_launcher = None
            logger.info("CloudXR runtime stopped")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not yet implemented.

    def _running_events(self) -> ExecutionEvents:
        """Constant ``RUNNING`` ``ExecutionEvents`` — the clutch lifecycle is owned by the loop."""
        return ExecutionEvents(execution_state=ExecutionState.RUNNING, reset=False)

    def _step(
        self,
        *,
        execution_events: ExecutionEvents | None = None,
        external_inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self._session.step(
            execution_events=execution_events,
            external_inputs=external_inputs,
        )
        info = self._session.last_step_info
        if info is not None:
            if info.worker_exception is not None:
                raise RuntimeError(
                    "Isaac Teleop retargeting worker raised an exception"
                ) from info.worker_exception
            if info.frame_deadline_miss:
                logger.warning(
                    "Isaac Teleop frame deadline miss (returned_age_frames=%s)",
                    info.returned_age_frames,
                )
        return result


# ======================================================================================
# XR controller
# ======================================================================================


class XRController(IsaacTeleopTeleoperator):
    """Raw XR controller grip-pose teleoperator (base-frame), no retargeters."""

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig):
        super().__init__(config)
        self.config: XRControllerConfig = config
        self._external_inputs: dict[str, Any] | None = None
        self._is_tracking = False

    def _build_pipeline(self) -> OutputCombiner:
        side = self.config.hand_side
        controller_key = f"controller_{side}"

        controllers = ControllersSource(name="controllers")
        xform = ValueInput(_BASE_T_ANCHOR_INPUT, TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        ctrl = transformed.output(controller_key)

        return OutputCombiner({"controller": ctrl})

    def _build_external_inputs(self) -> dict[str, Any]:
        """Materialize the constant ``base_T_anchor`` external input (once, in connect)."""
        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        return {_BASE_T_ANCHOR_INPUT: {"value": tg}}

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        try:
            self._external_inputs = self._build_external_inputs()
        except Exception:
            # Roll the session/runtime back so a failed connect() leaves no half-state
            # (a live session behind a raised connect would leak the CloudXR runtime).
            self.disconnect()
            raise

    @property
    def action_features(self) -> dict:
        return {
            "grip_pos": {"dtype": "float32", "shape": (3,)},
            "grip_quat": {"dtype": "float32", "shape": (4,)},
            "squeeze": {"dtype": "float32", "shape": ()},
            "trigger": {"dtype": "float32", "shape": ()},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_tracking(self) -> bool:
        """Whether the last :meth:`get_action` read a tracked controller."""
        return self._is_tracking

    def get_action(self) -> RobotAction:
        """Step the session and return the raw base-frame grip pose + squeeze/trigger."""
        result = self._step(execution_events=self._running_events(), external_inputs=self._external_inputs)

        controller = result["controller"]
        grip_pos = np.zeros(3, dtype=np.float32)
        grip_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        squeeze = 0.0
        trigger = 0.0
        self._is_tracking = not getattr(controller, "is_none", False)
        if self._is_tracking:
            # Read ALL four fields into locals before committing any of them: a failure on a
            # partially-populated frame must not mix live values with the safe defaults (a
            # live squeeze paired with a defaulted trigger=0.0 would keep the clutch engaged
            # while commanding the gripper fully open, dropping whatever is grasped). On
            # failure the defaults stand untouched and the frame reports not-tracked.
            try:
                grip_is_valid = bool(controller[ControllerInputIndex.GRIP_IS_VALID])
                pos = np.asarray(controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32)
                quat = np.asarray(controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float32)
                squeeze_val = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
                trigger_val = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            except (IndexError, KeyError, TypeError, ValueError):
                self._is_tracking = False
            else:
                quat_norm = float(np.linalg.norm(quat))
                if (
                    not grip_is_valid
                    or pos.shape != (3,)
                    or quat.shape != (4,)
                    or not np.all(np.isfinite(pos))
                    or not np.all(np.isfinite(quat))
                    or not np.isfinite(squeeze_val)
                    or not np.isfinite(trigger_val)
                    or quat_norm <= 1e-6
                ):
                    self._is_tracking = False
                else:
                    grip_pos, grip_quat = pos, quat / quat_norm
                    squeeze, trigger = squeeze_val, trigger_val

        return {
            "grip_pos": grip_pos,
            "grip_quat": grip_quat,
            "squeeze": squeeze,
            "trigger": trigger,
        }


# ======================================================================================
# Pose math
# ======================================================================================


def _pose6_to_base_t_ee(pose: tuple[float, float, float, float, float, float]) -> np.ndarray:
    """Convert a JAKA ``(x, y, z, roll, pitch, yaw)`` pose to a 4x4 base_T_ee matrix.

    XYZ-Euler radians, intrinsic rotations: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    x, y, z, roll, pitch, yaw = (float(v) for v in pose)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
    out = np.eye(4, dtype=float)
    out[:3, :3] = rot
    out[:3, 3] = np.array([x, y, z], dtype=float)
    return out


def _matrix_to_rpy(
    matrix: np.ndarray, *, reference_rpy: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Decompose a matrix into a JAKA-continuous ``(roll, pitch, yaw)`` triple."""
    return tuple(matrix_to_rpy(matrix, reference_rpy=reference_rpy))


# ======================================================================================
# Network helpers (CloudXR connect hint)
# ======================================================================================


def _primary_ipv4() -> str | None:
    """The workstation's primary outbound IPv4 via the UDP-socket trick."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return None


def _candidate_ipv4s() -> list[tuple[str, str]]:
    """Return ``[(interface, ipv4), ...]`` the headset might reach this workstation at.

    Primary outbound first; drops loopback, link-local, and virtual/bridge interfaces.
    """
    primary = _primary_ipv4()
    found: list[tuple[str, str]] = []
    try:
        import psutil

        for iface, addrs in psutil.net_if_addrs().items():
            if iface.startswith(_SKIP_IFACE_PREFIXES):
                continue
            for addr in addrs:
                if addr.family != socket.AF_INET:
                    continue
                ip = addr.address
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                found.append((iface, ip))
    except Exception:
        if primary:
            found.append(("default", primary))
    found.sort(key=lambda t: t[1] != primary)
    return found


def _print_xr_connect_help() -> None:
    """Print how to connect the headset to this workstation over CloudXR."""
    ips = _candidate_ipv4s()
    print("\n" + "=" * 76)
    print("Connect your XR headset to this workstation over NVIDIA CloudXR:")
    print(f"  1. In the headset, open the CloudXR web client:  {_CLOUDXR_WEB_CLIENT_URL}")
    print("  2. Enter this workstation's IP address:")
    if ips:
        for iface, ip in ips:
            print(f"        {ip:<15}  ({iface})")
        if len(ips) > 1:
            print("     (use the address on the same network as your headset)")
    else:
        print("        <could not determine — check `hostname -I` / `ip addr`>")
    print(f"  3. Accept the self-signed cert at https://<that-ip>:{_CLOUDXR_WSS_PORT}/ , then Connect.")
    print("=" * 76 + "\n")


def _wait_for_xr_controller(teleop: XRController) -> None:
    """Block until the XR controller is tracked."""
    _print_xr_connect_help()
    print("Waiting for the headset controllers to start streaming…  (Ctrl-C to abort)")
    last_reminder = time.time()
    while True:
        teleop.get_action()  # steps the session; updates is_tracking
        if teleop.is_tracking:
            print("Headset connected — controllers are streaming.")
            return
        if time.time() - last_reminder >= _CONNECT_REMINDER_S:
            print("…still waiting for the headset to connect (Ctrl-C to abort).")
            last_reminder = time.time()
        time.sleep(1.0 / POLL_HZ)


# ======================================================================================
# JAKA device bundle
# ======================================================================================


def make_xr_device(robot, teleop_config: XRControllerConfig) -> dict:
    """Build a (startup, compute, cleanup, telemetry) bundle for the JAKA recorder.

    ``compute(obs)`` returns ``None`` while the clutch is disengaged (the recorder's
    HoldLatch keeps the arm on the measured pose); otherwise an absolute Cartesian RPY
    action matching :attr:`JakaRobot.action_features`.

    The JAKA driver (``JakaRobot._send_eef_action``) applies its own per-frame Cartesian
    step limit and Servo Move streaming — this device only produces the target.
    """
    teleop = XRController(teleop_config)
    clutch: Clutch | None = None
    prev_enabled = False
    last_pos: np.ndarray | None = None
    last_rpy: np.ndarray | None = None
    locked_rpy: np.ndarray | None = None
    telemetry: dict[str, object] = {}

    def startup() -> None:
        nonlocal clutch
        teleop.connect()
        if not teleop.is_connected:
            raise ValueError("Teleop is not connected!")
        _wait_for_xr_controller(teleop)

        # Seed the clutch home from the arm's measured EE pose so the first engage
        # is jump-free.
        measured_pose = tuple(float(v) for v in robot.get_eef_pose())
        home_base_t_ee = _pose6_to_base_t_ee(measured_pose)
        clutch = Clutch(home_base_t_ee)

        if robot.name == "jaka_robot":
            robot.servo_enable(True)

        print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")

    def compute(robot_obs) -> dict | None:
        nonlocal clutch, prev_enabled, last_pos, last_rpy, locked_rpy
        if clutch is None:
            raise RuntimeError("compute() called before startup()")

        xr_action = teleop.get_action()
        grip_pos = np.asarray(xr_action["grip_pos"], dtype=float)
        grip_quat = np.asarray(xr_action["grip_quat"], dtype=float)
        squeeze = float(xr_action["squeeze"])
        trigger = float(xr_action["trigger"])
        enabled = squeeze > teleop_config.clutch_threshold
        telemetry.update(
            grip_pos=tuple(float(v) for v in grip_pos),
            grip_quat=tuple(float(v) for v in grip_quat),
            squeeze=squeeze,
            trigger=trigger,
            clutch_engaged=enabled,
        )

        # On the engage edge, latch the clutch home at the arm's MEASURED EE pose so
        # the first engaged frame commands zero delta. Latching the last commanded pose
        # would snap the arm back at full servo speed if it moved while disengaged
        # (gravity sag, external contact).
        is_engage_frame = enabled and not prev_enabled
        if is_engage_frame:
            eef_keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            if robot_obs is None or not all(key in robot_obs for key in eef_keys):
                raise RuntimeError("XR clutch engage requires a complete measured EEF observation")
            measured_pose = tuple(float(robot_obs[key]) for key in eef_keys)
            measured_base_t_ee = _pose6_to_base_t_ee(measured_pose)
            clutch.engage(grip_pos, grip_quat, home_base_T_ee=measured_base_t_ee)
            last_pos = None  # drop the rate-limit reference so we don't fight the new home
            last_rpy = np.asarray(measured_pose[3:], dtype=float)
            locked_rpy = last_rpy.copy() if teleop_config.lock_pose else None
        prev_enabled = enabled

        # Hold the arm at the measured pose while the clutch is disengaged — the
        # recorder's HoldLatch freezes the value on the first idle frame.
        if not enabled:
            last_rpy = None
            locked_rpy = None
            return None

        pos, quat = clutch.rebase(grip_pos, grip_quat)

        # Per-frame position step limit (defence in depth; JAKA's own clamp is stricter).
        if last_pos is not None:
            delta = pos - last_pos
            n = float(np.linalg.norm(delta))
            if n > MAX_EE_STEP_M:
                pos = last_pos + delta * (MAX_EE_STEP_M / n)
        last_pos = pos

        if last_rpy is None:
            raise RuntimeError("XR orientation state was not initialized on clutch engage")
        if teleop_config.lock_pose:
            if locked_rpy is None:
                raise RuntimeError("XR locked orientation was not initialized on clutch engage")
            roll, pitch, yaw = (float(value) for value in locked_rpy)
        else:
            roll, pitch, yaw = _matrix_to_rpy(Rotation.from_quat(quat).as_matrix(), reference_rpy=last_rpy)
            last_rpy = np.array([roll, pitch, yaw])
        return {
            "ee.x": float(pos[0]),
            "ee.y": float(pos[1]),
            "ee.z": float(pos[2]),
            "ee.roll": roll,
            "ee.pitch": pitch,
            "ee.yaw": yaw,
            "gripper.pos": float(np.clip(1.0 - trigger, 0.0, 1.0)),
        }

    def cleanup() -> None:
        try:
            if robot.name == "jaka_robot" and robot.is_connected and robot.is_in_servo():
                robot.servo_enable(False)
        finally:
            teleop.disconnect()

    return {
        "startup": startup,
        "compute": compute,
        "cleanup": cleanup,
        "telemetry": telemetry,
    }


__all__ = [
    "CLOUDXR_ENV_FILE",
    "Clutch",
    "IsaacTeleopConfig",
    "IsaacTeleopTeleoperator",
    "MAX_EE_STEP_M",
    "POLL_HZ",
    "XRController",
    "XRControllerConfig",
    "make_xr_device",
]
