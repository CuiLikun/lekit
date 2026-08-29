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
import math
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from lekit.robots.jaka_robot.pose_math import matrix_to_rpy
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.import_utils import is_package_available
from lerobot.utils.rotation import Rotation

from .clutch import Clutch

logger = logging.getLogger(__name__)

# Fixed poll rate for the pre-loop waits. Same value the original common module used; only
# consumed by the connect-wait polling, not the control loop.
POLL_HZ = 30

# Max per-frame EE translation step [m/frame]. Guards the rotvec->RPY path against a
# tracking glitch producing a one-frame teleport. JAKA's own per-frame clamp is stricter
# (0.01 m); this is defence in depth.
MAX_EE_STEP_M = 0.05

# Bound the orientation integrator's elapsed time so a stalled control loop cannot
# turn one held thumbstick sample into a large TCP orientation jump.
MAX_THUMBSTICK_DT_S = 0.1

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
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HeadSource
    from isaacteleop.retargeting_engine.interface import (
        ExecutionEvents,
        ExecutionState,
        OutputCombiner,
        TensorGroup,
        ValueInput,
    )
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
    from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex, HeadPoseIndex
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig
else:
    CloudXRLauncher = None
    ControllersSource = None
    HeadSource = None
    ExecutionEvents = None
    ExecutionState = None
    OutputCombiner = None
    TensorGroup = None
    ValueInput = None
    TransformMatrix = None
    ControllerInputIndex = None
    HeadPoseIndex = None
    TeleopSession = None
    TeleopSessionConfig = None

# Static rebase from the CloudXR controller anchor frame (X=Right, Y=Up, Z=Backward)
# into the JAKA base frame (X=Right, Y=Forward, Z=Up). Head-relative control latches the
# operator's horizontal heading on each clutch engage; ``operator_yaw_deg`` is an optional
# correction between the headset's gaze direction and the operator's intended forward.
DEFAULT_OPERATOR_YAW_DEG = 0.0


def _base_t_anchor_for_yaw(yaw_deg: float) -> list[list[float]]:
    """Build the OpenXR -> JAKA rebase for an operator station yawed CCW (from above)
    by ``yaw_deg`` around the vertical axis.

    At ``yaw_deg=0`` hand right maps to robot +X, hand forward maps to robot +Y,
    and hand up maps to robot +Z.
    """
    yaw_rad = np.deg2rad(float(yaw_deg))
    yaw_sin = float(np.sin(yaw_rad))
    yaw_cos = float(np.cos(yaw_rad))
    return [
        [yaw_cos, 0.0, yaw_sin, 0.0],
        [yaw_sin, 0.0, -yaw_cos, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _base_t_anchor_from_head_quat(
    head_quat: np.ndarray, *, yaw_offset_deg: float = 0.0
) -> np.ndarray:
    """Build an OpenXR-anchor -> JAKA-base transform from the headset's horizontal yaw."""

    quat = np.asarray(head_quat, dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("head quaternion must contain four finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-6:
        raise ValueError("head quaternion norm is too small")

    anchor_r_head = Rotation.from_quat(quat / norm).as_matrix()
    forward = anchor_r_head @ np.array([0.0, 0.0, -1.0])
    horizontal_norm = float(np.linalg.norm(forward[[0, 2]]))
    if horizontal_norm <= 1e-3:
        raise ValueError("head forward direction is too close to vertical")
    forward /= horizontal_norm
    heading_deg = float(np.rad2deg(np.arctan2(forward[0], -forward[2])))
    return np.asarray(_base_t_anchor_for_yaw(heading_deg + yaw_offset_deg), dtype=float)


def _transform_grip_pose(
    position: np.ndarray, quaternion: np.ndarray, base_t_anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an anchor-to-base transform to an OpenXR grip pose."""

    transform = np.asarray(base_t_anchor, dtype=float)
    rotation = transform[:3, :3]
    transformed_position = rotation @ np.asarray(position, dtype=float) + transform[:3, 3]
    transformed_rotation = Rotation.from_matrix(rotation) * Rotation.from_quat(
        np.asarray(quaternion, dtype=float)
    )
    return transformed_position, transformed_rotation.as_quat()


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
    """Use measured JAKA roll/pitch/yaw as the thumbstick-adjustable orientation base."""

    thumbstick_deadband: float = 0.15
    """Ignore centered thumbstick input below this normalized magnitude."""

    thumbstick_angular_speed_rad_s: float = 0.5
    """Maximum per-axis TCP orientation rate from a fully deflected thumbstick."""

    tool_tip_offset_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Tool-0 flange to physical pivot point in tool-local XYZ metres."""

    position_deadband_m: float = 0.0002
    """Ignore Cartesian controller drift up to this distance from the last target."""

    operator_yaw_deg: float = DEFAULT_OPERATOR_YAW_DEG
    """Horizontal correction in degrees from headset gaze to intended operator forward."""

    use_head_yaw: bool = True
    """Latch headset yaw on each clutch engage so translation follows operator heading."""

    servo_linear_velocity_m_s: float = 0.2
    """Maximum linear Servo P velocity used by the teleoperation profile."""

    servo_linear_acceleration_m_s2: float = 0.8
    """Maximum linear Servo P acceleration used by the teleoperation profile."""

    servo_linear_jerk_m_s3: float = 3.0
    """Cartesian NLF linear jerk used by the teleoperation profile."""

    servo_angular_velocity_rad_s: float = 1.0
    """Maximum angular Servo P velocity used by the teleoperation profile."""

    servo_angular_acceleration_rad_s2: float = 2.0
    """Maximum angular Servo P acceleration used by the teleoperation profile."""

    servo_angular_jerk_rad_s3: float = 8.0
    """Cartesian NLF angular jerk used by the teleoperation profile."""

    base_T_anchor: list[list[float]] | None = field(default=None)  # noqa: N815
    """Static 4x4 transform rebasing the OpenXR anchor frame into the robot base frame."""

    def __post_init__(self):
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {self.hand_side!r}")
        if not isinstance(self.lock_pose, bool):
            raise ValueError("lock_pose must be a boolean")
        if (
            not np.isfinite(self.thumbstick_deadband)
            or not 0.0 <= self.thumbstick_deadband < 1.0
        ):
            raise ValueError("thumbstick_deadband must be finite and in [0, 1)")
        if (
            not np.isfinite(self.thumbstick_angular_speed_rad_s)
            or self.thumbstick_angular_speed_rad_s <= 0.0
        ):
            raise ValueError("thumbstick_angular_speed_rad_s must be positive and finite")
        tool_tip_offset = np.asarray(self.tool_tip_offset_m, dtype=float)
        if tool_tip_offset.shape != (3,) or not np.all(np.isfinite(tool_tip_offset)):
            raise ValueError("tool_tip_offset_m must contain exactly three finite XYZ values")
        self.tool_tip_offset_m = [float(value) for value in tool_tip_offset]
        if not np.isfinite(self.position_deadband_m) or self.position_deadband_m < 0:
            raise ValueError("position_deadband_m must be finite and non-negative")
        if not np.isfinite(self.operator_yaw_deg):
            raise ValueError("operator_yaw_deg must be finite")
        if not isinstance(self.use_head_yaw, bool):
            raise ValueError("use_head_yaw must be a boolean")
        if self.base_T_anchor is None:
            self.base_T_anchor = _base_t_anchor_for_yaw(self.operator_yaw_deg)
        for name in (
            "servo_linear_velocity_m_s",
            "servo_linear_acceleration_m_s2",
            "servo_linear_jerk_m_s3",
            "servo_angular_velocity_rad_s",
            "servo_angular_acceleration_rad_s2",
            "servo_angular_jerk_rad_s3",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


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
        head = HeadSource(name="head")
        raw_ctrl = controllers.output(controller_key)
        xform = ValueInput(_BASE_T_ANCHOR_INPUT, TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        ctrl = transformed.output(controller_key)

        return OutputCombiner(
            {"controller": ctrl, "controller_raw": raw_ctrl, "head": head.output("head")}
        )

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
            "raw_grip_pos": {"dtype": "float32", "shape": (3,)},
            "raw_grip_quat": {"dtype": "float32", "shape": (4,)},
            "head_quat": {"dtype": "float32", "shape": (4,)},
            "head_is_tracking": {"dtype": "bool", "shape": ()},
            "squeeze": {"dtype": "float32", "shape": ()},
            "trigger": {"dtype": "float32", "shape": ()},
            "a_button": {"dtype": "float32", "shape": ()},
            "b_button": {"dtype": "float32", "shape": ()},
            "thumbstick_x": {"dtype": "float32", "shape": ()},
            "thumbstick_y": {"dtype": "float32", "shape": ()},
            "thumbstick_click": {"dtype": "float32", "shape": ()},
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
        a_button = 0.0
        b_button = 0.0
        thumbstick_x = 0.0
        thumbstick_y = 0.0
        thumbstick_click = 0.0
        self._is_tracking = not getattr(controller, "is_none", False)
        if self._is_tracking:
            # Read the pose/deadman fields into locals before committing any of them: a failure
            # on a partially-populated frame must not mix live values with the safe defaults (a
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
                    try:
                        a_val = float(controller[ControllerInputIndex.PRIMARY_CLICK])
                        b_val = float(controller[ControllerInputIndex.SECONDARY_CLICK])
                    except (IndexError, KeyError, TypeError, ValueError):
                        pass
                    else:
                        if np.isfinite(a_val):
                            a_button = a_val
                        if np.isfinite(b_val):
                            b_button = b_val
                    try:
                        stick_x_val = float(controller[ControllerInputIndex.THUMBSTICK_X])
                        stick_y_val = float(controller[ControllerInputIndex.THUMBSTICK_Y])
                        stick_click_val = float(
                            controller[ControllerInputIndex.THUMBSTICK_CLICK]
                        )
                    except (IndexError, KeyError, TypeError, ValueError):
                        pass
                    else:
                        if np.isfinite(stick_x_val):
                            thumbstick_x = float(np.clip(stick_x_val, -1.0, 1.0))
                        if np.isfinite(stick_y_val):
                            thumbstick_y = float(np.clip(stick_y_val, -1.0, 1.0))
                        if np.isfinite(stick_click_val):
                            thumbstick_click = float(np.clip(stick_click_val, 0.0, 1.0))

        raw_grip_pos = grip_pos.copy()
        raw_grip_quat = grip_quat.copy()
        raw_controller = result.get("controller_raw")
        if self._is_tracking and raw_controller is not None and not getattr(raw_controller, "is_none", False):
            try:
                raw_is_valid = bool(raw_controller[ControllerInputIndex.GRIP_IS_VALID])
                raw_pos = np.asarray(raw_controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32)
                raw_quat = np.asarray(raw_controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float32)
            except (IndexError, KeyError, TypeError, ValueError):
                pass
            else:
                raw_quat_norm = float(np.linalg.norm(raw_quat))
                if (
                    raw_is_valid
                    and raw_pos.shape == (3,)
                    and raw_quat.shape == (4,)
                    and np.all(np.isfinite(raw_pos))
                    and np.all(np.isfinite(raw_quat))
                    and raw_quat_norm > 1e-6
                ):
                    raw_grip_pos = raw_pos
                    raw_grip_quat = raw_quat / raw_quat_norm

        head_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        head_is_tracking = False
        head = result.get("head")
        if head is not None and not getattr(head, "is_none", False):
            try:
                head_is_valid = bool(head[HeadPoseIndex.IS_VALID])
                candidate = np.asarray(head[HeadPoseIndex.ORIENTATION], dtype=np.float32)
            except (IndexError, KeyError, TypeError, ValueError):
                pass
            else:
                candidate_norm = float(np.linalg.norm(candidate))
                if (
                    head_is_valid
                    and candidate.shape == (4,)
                    and np.all(np.isfinite(candidate))
                    and candidate_norm > 1e-6
                ):
                    head_quat = candidate / candidate_norm
                    head_is_tracking = True

        return {
            "grip_pos": grip_pos,
            "grip_quat": grip_quat,
            "raw_grip_pos": raw_grip_pos,
            "raw_grip_quat": raw_grip_quat,
            "head_quat": head_quat,
            "head_is_tracking": head_is_tracking,
            "squeeze": squeeze,
            "trigger": trigger,
            "a_button": a_button,
            "b_button": b_button,
            "thumbstick_x": thumbstick_x,
            "thumbstick_y": thumbstick_y,
            "thumbstick_click": thumbstick_click,
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


def _thumbstick_axis(value: float, deadband: float) -> float:
    """Apply a continuous deadband while preserving full-scale stick output."""

    value = float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else 0.0
    magnitude = abs(value)
    if magnitude <= deadband:
        return 0.0
    return math.copysign((magnitude - deadband) / (1.0 - deadband), value)


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
    last_clutch_pos: np.ndarray | None = None
    last_pos: np.ndarray | None = None
    last_rpy: np.ndarray | None = None
    locked_rpy: np.ndarray | None = None
    thumbstick_rotation = Rotation.from_rotvec(np.zeros(3, dtype=float))
    thumbstick_updated_at: float | None = None
    latched_base_t_anchor: np.ndarray | None = None
    telemetry: dict[str, object] = {}
    tool_tip_offset = np.asarray(teleop_config.tool_tip_offset_m, dtype=float)

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

        print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")

    def compute(robot_obs) -> dict | None:
        nonlocal clutch, prev_enabled, last_clutch_pos, last_pos, last_rpy, locked_rpy
        nonlocal latched_base_t_anchor, thumbstick_rotation, thumbstick_updated_at
        if clutch is None:
            raise RuntimeError("compute() called before startup()")

        xr_action = teleop.get_action()
        frame_time = time.monotonic()
        grip_pos = np.asarray(xr_action["grip_pos"], dtype=float)
        grip_quat = np.asarray(xr_action["grip_quat"], dtype=float)
        raw_grip_pos = np.asarray(xr_action.get("raw_grip_pos", grip_pos), dtype=float)
        raw_grip_quat = np.asarray(xr_action.get("raw_grip_quat", grip_quat), dtype=float)
        head_quat = np.asarray(
            xr_action.get("head_quat", [0.0, 0.0, 0.0, 1.0]), dtype=float
        )
        head_is_tracking = bool(xr_action.get("head_is_tracking", False))
        squeeze = float(xr_action["squeeze"])
        trigger = float(xr_action["trigger"])
        a_button = float(xr_action.get("a_button", 0.0))
        b_button = float(xr_action.get("b_button", 0.0))
        thumbstick_x = float(xr_action.get("thumbstick_x", 0.0))
        thumbstick_y = float(xr_action.get("thumbstick_y", 0.0))
        thumbstick_click = float(xr_action.get("thumbstick_click", 0.0))
        requested_enabled = squeeze > teleop_config.clutch_threshold

        # Use the operator's heading at the engage edge, then keep that transform fixed
        # until release. Continuous head following would move a stationary hand whenever
        # the operator merely looks sideways.
        if requested_enabled and not prev_enabled and teleop_config.use_head_yaw:
            if head_is_tracking:
                try:
                    latched_base_t_anchor = _base_t_anchor_from_head_quat(
                        head_quat, yaw_offset_deg=teleop_config.operator_yaw_deg
                    )
                except ValueError:
                    latched_base_t_anchor = None
            else:
                latched_base_t_anchor = None

        enabled = requested_enabled and (
            not teleop_config.use_head_yaw or latched_base_t_anchor is not None
        )
        is_release_frame = not enabled and prev_enabled
        if teleop_config.use_head_yaw and latched_base_t_anchor is not None:
            grip_pos, grip_quat = _transform_grip_pose(
                raw_grip_pos, raw_grip_quat, latched_base_t_anchor
            )

        control_yaw_deg = None
        if latched_base_t_anchor is not None:
            control_yaw_deg = float(
                np.rad2deg(
                    np.arctan2(latched_base_t_anchor[0, 2], latched_base_t_anchor[0, 0])
                )
            )
        telemetry.update(
            grip_pos=tuple(float(v) for v in grip_pos),
            grip_quat=tuple(float(v) for v in grip_quat),
            raw_grip_pos=tuple(float(v) for v in raw_grip_pos),
            raw_grip_quat=tuple(float(v) for v in raw_grip_quat),
            head_quat=tuple(float(v) for v in head_quat),
            head_is_tracking=head_is_tracking,
            controller_is_tracking=bool(getattr(teleop, "is_tracking", True)),
            control_yaw_deg=control_yaw_deg,
            squeeze=squeeze,
            trigger=trigger,
            a_button=a_button,
            b_button=b_button,
            thumbstick_x=thumbstick_x,
            thumbstick_y=thumbstick_y,
            thumbstick_click=thumbstick_click,
            clutch_engaged=enabled,
            clutch_released=is_release_frame,
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
            last_clutch_pos = None
            last_pos = None  # drop the rate-limit reference so we don't fight the new home
            last_rpy = np.asarray(measured_pose[3:], dtype=float)
            locked_rpy = last_rpy.copy() if teleop_config.lock_pose else None
            thumbstick_rotation = Rotation.from_rotvec(np.zeros(3, dtype=float))
            thumbstick_updated_at = frame_time
        prev_enabled = enabled

        # Hold the arm at the measured pose while the clutch is disengaged — the
        # recorder's HoldLatch freezes the value on the first idle frame.
        if not enabled:
            last_clutch_pos = None
            last_pos = None
            last_rpy = None
            locked_rpy = None
            thumbstick_rotation = Rotation.from_rotvec(np.zeros(3, dtype=float))
            thumbstick_updated_at = None
            if not requested_enabled:
                latched_base_t_anchor = None
            return None

        pos, quat = clutch.rebase(grip_pos, grip_quat)

        # Filter controller translation before adding the deterministic pivot compensation.
        if last_clutch_pos is not None:
            delta = pos - last_clutch_pos
            n = float(np.linalg.norm(delta))
            if n <= teleop_config.position_deadband_m:
                pos = last_clutch_pos.copy()
            elif n > MAX_EE_STEP_M:
                pos = last_clutch_pos + delta * (MAX_EE_STEP_M / n)
        last_clutch_pos = pos.copy()

        if last_rpy is None:
            raise RuntimeError("XR orientation state was not initialized on clutch engage")
        elapsed_s = (
            min(max(frame_time - thumbstick_updated_at, 0.0), MAX_THUMBSTICK_DT_S)
            if thumbstick_updated_at is not None
            else 0.0
        )
        thumbstick_updated_at = frame_time
        stick_x = _thumbstick_axis(thumbstick_x, teleop_config.thumbstick_deadband)
        stick_y = _thumbstick_axis(thumbstick_y, teleop_config.thumbstick_deadband)
        angular_step = teleop_config.thumbstick_angular_speed_rad_s * elapsed_s
        # Base-frame axes keep the mapping operator-relative: +X rotation tilts a
        # downward tool forward, while -Y rotation tilts it right. Stick-click keeps
        # the third degree of freedom available as base-frame yaw.
        if thumbstick_click >= 0.5:
            incremental_rotvec = np.array(
                [stick_y * angular_step, 0.0, stick_x * angular_step], dtype=float
            )
        else:
            incremental_rotvec = np.array(
                [stick_y * angular_step, -stick_x * angular_step, 0.0], dtype=float
            )
        thumbstick_rotation = Rotation.from_rotvec(incremental_rotvec) * thumbstick_rotation

        if teleop_config.lock_pose:
            if locked_rpy is None:
                raise RuntimeError("XR locked orientation was not initialized on clutch engage")
            locked_pose = (0.0, 0.0, 0.0, *(float(value) for value in locked_rpy))
            base_rotation = Rotation.from_matrix(_pose6_to_base_t_ee(locked_pose)[:3, :3])
        else:
            base_rotation = Rotation.from_quat(quat)
        target_rotation = thumbstick_rotation * base_rotation
        target_rpy = np.asarray(
            _matrix_to_rpy(target_rotation.as_matrix(), reference_rpy=last_rpy), dtype=float
        )
        last_rpy = target_rpy.copy()

        # Keep the physical tool tip fixed while orientation changes. ``pos`` is the
        # uncompensated tool-0 flange target; both offset terms are expressed in base.
        pos = (
            pos
            + base_rotation.as_matrix() @ tool_tip_offset
            - target_rotation.as_matrix() @ tool_tip_offset
        )

        # Apply the defence-in-depth Cartesian step limit to the final flange command.
        if last_pos is not None:
            delta = pos - last_pos
            n = float(np.linalg.norm(delta))
            if n > MAX_EE_STEP_M:
                pos = last_pos + delta * (MAX_EE_STEP_M / n)
        last_pos = pos.copy()

        roll, pitch, yaw = (float(value) for value in target_rpy)
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

    def rearm() -> None:
        """Force the next squeeze frame to latch a fresh measured robot pose."""

        nonlocal prev_enabled, last_clutch_pos, last_pos, last_rpy, locked_rpy
        nonlocal latched_base_t_anchor, thumbstick_rotation, thumbstick_updated_at
        prev_enabled = False
        last_clutch_pos = None
        last_pos = None
        last_rpy = None
        locked_rpy = None
        latched_base_t_anchor = None
        thumbstick_rotation = Rotation.from_rotvec(np.zeros(3, dtype=float))
        thumbstick_updated_at = None
        telemetry.update(clutch_engaged=False, clutch_released=True)

    return {
        "startup": startup,
        "compute": compute,
        "cleanup": cleanup,
        "rearm": rearm,
        "telemetry": telemetry,
    }


__all__ = [
    "CLOUDXR_ENV_FILE",
    "Clutch",
    "IsaacTeleopConfig",
    "IsaacTeleopTeleoperator",
    "MAX_EE_STEP_M",
    "MAX_THUMBSTICK_DT_S",
    "POLL_HZ",
    "XRController",
    "XRControllerConfig",
    "make_xr_device",
]
