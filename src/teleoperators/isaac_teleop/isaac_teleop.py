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

"""Standard Isaac Teleop teleoperator.

Defines:

  * :class:`IsaacTeleopConfig` — a :class:`TeleoperatorConfig` registered as
    ``--teleop.type=isaac_teleop`` so draccus can decode the CLI.
  * :class:`IsaacTeleop` — a :class:`Teleoperator` that reads raw XR controller
    grip pose (base-frame), squeeze, and trigger via an Isaac Teleop session.

Usage::

    teleop = IsaacTeleop(IsaacTeleopConfig())
    teleop.connect()
    while True:
        action = teleop.get_action()
        # action == {"grip_pos": ..., "grip_quat": ..., "squeeze": ..., "trigger": ...}
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
from scipy.spatial.transform import Rotation

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

logger = logging.getLogger(__name__)

# Default static rebase from OpenXR anchor (X=Right, Y=Up, Z=Backward)
# to robot base (X=Forward, Y=Left, Z=Up).
_DEFAULT_BASE_T_ANCHOR: list[list[float]] = [
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


@TeleoperatorConfig.register_subclass("isaac_teleop")
@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Config for the standard Isaac Teleop teleoperator.

    Exposes the raw controller grip pose, squeeze, and trigger via
    ``ControllersSource`` — no retargeters. The clutch and gripper mapping
    live in the owning loop.
    """

    app_name: str = "LeTeleop"
    """Application name for the OpenXR / Isaac Teleop session."""

    auto_launch_cloudxr: bool = True
    """Auto-launch the CloudXR runtime on :meth:`connect`. Set ``False`` (or
    ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1``) when CloudXR runs externally."""

    cloudxr_env_file: str | None = None
    """Optional CloudXR device-profile ``.env`` passed to ``CloudXRLauncher``."""

    hand_side: str = "right"
    """Controller hand: ``"left"`` or ``"right"``."""

    base_T_anchor: list[list[float]] = field(  # noqa: N815
        default_factory=lambda: [row.copy() for row in _DEFAULT_BASE_T_ANCHOR]
    )
    """Static 4x4 transform rebasing the OpenXR controller anchor frame into
    the robot base frame. Row-major."""

    def __post_init__(self):
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {self.hand_side!r}")


class IsaacTeleop(Teleoperator):
    """Standard Isaac Teleop teleoperator — raw XR controller reader.

    Builds a minimal pipeline (``ControllersSource`` → frame rebase →
    ``OutputCombiner``) and exposes the absolute base-frame grip pose, squeeze,
    and trigger. No retargeters, no clutch — those live in the owning loop.
    """

    config_class = IsaacTeleopConfig
    name = "isaac_teleop"

    # Source-node name for the static base_T_anchor input fed via
    # ``TeleopSession.step(external_inputs=...)`` each frame.
    _BASE_T_ANCHOR_INPUT = "base_T_anchor"

    # Squeeze hysteresis for the clutch state machine. ``squeeze`` must climb
    # above the engage threshold to latch an origin, then drop below the
    # release threshold to free it — prevents chatter at the boundary.
    _CLUTCH_ENGAGE_THRESHOLD = 0.5
    _CLUTCH_RELEASE_THRESHOLD = 0.3

    def __init__(self, config: IsaacTeleopConfig):
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self._session: TeleopSession | None = None
        self._cloudxr_launcher: CloudXRLauncher | None = None
        self._external_inputs: dict[str, Any] | None = None
        self._is_tracking = False

        # Clutch (squeeze-engaged relative-motion) state. ``_clutch_prev_*``
        # holds the previous frame's raw pose. The reported ``grip_pos`` is
        # the per-frame increment ``raw - prev`` (so the consumer can do
        # ``ee += grip_pos`` without any other state). On the rising edge of
        # squeeze we set ``prev = raw`` so the first engaged frame yields a
        # zero increment — no jump.
        self._clutch_engaged: bool = False
        self._clutch_prev_pos: np.ndarray | None = None
        self._clutch_prev_quat: np.ndarray | None = None  # unit (xyzw)

    # ---------------------------------------------------------------- features

    @property
    def action_features(self) -> dict:
        return {
            "grip_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": {"x": 0, "y": 1, "z": 2},
            },
            "grip_quat": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"qx": 0, "qy": 1, "qz": 2, "qw": 3},
            },
            "grip_ori": {
                "dtype": "float32",
                "shape": (3,),
                "names": {"roll": 0, "pitch": 1, "yaw": 2},
            },
            "squeeze": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "trigger": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "a_button": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "b_button": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    # --------------------------------------------------------------- lifecycle

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Tracking devices self-calibrate.

    @property
    def is_tracking(self) -> bool:
        """Whether the last :meth:`get_action` read a tracked controller."""
        return self._is_tracking

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, _feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not yet implemented.

    def connect(self, calibrate: bool = True) -> None:
        """Launch CloudXR (unless opted out), build the pipeline, and open the session.

        Retries until an XR system becomes available (i.e. a headset connects
        over CloudXR), printing connection instructions for the operator.
        """
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        if calibrate:
            self.calibrate()

        self._ensure_cloudxr_runtime()
        pipeline = self._build_pipeline()
        self._external_inputs = self._build_external_inputs()
        session_config = TeleopSessionConfig(app_name=self.config.app_name, pipeline=pipeline)

        self._print_connect_help()
        print("Waiting for XR system (connect your headset via CloudXR)…  (Ctrl-C to abort)")
        while True:
            try:
                self._session = TeleopSession(session_config)
                self._session.__enter__()
            except RuntimeError:
                self._session = None
                time.sleep(2.0)
            else:
                break
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def disconnect(self) -> None:
        """Tear down the session and (if we own it) the CloudXR runtime."""
        try:
            if self._session is not None:
                session = self._session
                self._session = None
                self._external_inputs = None
                session.__exit__(None, None, None)
                logger.info("Isaac Teleop session ended")
        finally:
            # Clutch state is per-session; clear so a fresh connect() does
            # not inherit stale prev/engaged state.
            self._clutch_engaged = False
            self._clutch_prev_pos = None
            self._clutch_prev_quat = None
            self._stop_cloudxr_runtime()

    # -------------------------------------------------------- CloudXR runtime

    # CloudXR web client URL and WSS port.
    _CLOUDXR_CLIENT_URL = "https://nvidia.github.io/IsaacTeleop/client"
    _CLOUDXR_WSS_PORT = 48322

    def _primary_ipv4(self) -> str | None:
        """Return the primary outbound IPv4 address."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
            except OSError:
                return None

    def _print_connect_help(self) -> None:
        """Print how to connect the headset to this workstation."""
        ip = self._primary_ipv4()
        print()
        print("=" * 72)
        print("Connect your XR headset to this workstation via CloudXR:")
        print(f"  1. In the headset browser, open:  {self._CLOUDXR_CLIENT_URL}")
        if ip:
            print(f"  2. Enter this workstation's IP:  {ip}")
        print(f"  3. Accept the self-signed cert at https://<ip>:{self._CLOUDXR_WSS_PORT}/ , then Connect.")
        print("=" * 72)
        print()

    def _ensure_cloudxr_runtime(self) -> None:
        """Auto-launch the CloudXR runtime once, unless opted out.

        Idempotent. ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH`` wins over
        ``config.auto_launch_cloudxr``.
        """
        if self._cloudxr_launcher is not None:
            return

        if os.environ.get("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "").strip() == "1":
            logger.info("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1 set; skipping CloudXR auto-launch.")
            return

        if not self.config.auto_launch_cloudxr:
            logger.info("config.auto_launch_cloudxr=False; skipping CloudXR auto-launch.")
            return

        logger.info("Launching CloudXR runtime (first run may prompt for EULA and take ~30s)...")
        self._cloudxr_launcher = CloudXRLauncher(
            install_dir=str(Path.home() / ".cloudxr"),
            env_config=self.config.cloudxr_env_file,
            accept_eula=False,
        )

    def _stop_cloudxr_runtime(self) -> None:
        """Stop the auto-launched CloudXR runtime, if any.

        On failure the handle is RETAINED so ``atexit`` owns the retry;
        a later ``connect()`` sees the retained runtime and won't relaunch.
        """
        if self._cloudxr_launcher is None:
            return
        try:
            self._cloudxr_launcher.stop()
        except RuntimeError:
            logger.warning("CloudXR runtime could not be terminated; handle retained for atexit cleanup")
        else:
            self._cloudxr_launcher = None
            logger.info("CloudXR runtime stopped")

    # ---------------------------------------------------------------- pipeline

    def _build_pipeline(self) -> OutputCombiner:
        """Build the minimal grip-pose pipeline: ``ControllersSource`` rebased
        into the base frame, exposed verbatim as ``"controller"``."""
        controllers = ControllersSource(name="controllers")
        xform = ValueInput(self._BASE_T_ANCHOR_INPUT, TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        controller_key = f"controller_{self.config.hand_side}"
        ctrl = transformed.output(controller_key)
        return OutputCombiner({"controller": ctrl})

    def _build_external_inputs(self) -> dict[str, Any]:
        """Materialize the constant ``base_T_anchor`` external input (once, in connect)."""
        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        return {self._BASE_T_ANCHOR_INPUT: {"value": tg}}

    # ------------------------------------------------------------------ step

    def _step(self) -> Any:
        """Step the session once and return the raw pipeline outputs.

        Guards: re-raises a retargeting-worker exception and warns on stale frames.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self._session.step(
            execution_events=ExecutionEvents(execution_state=ExecutionState.RUNNING, reset=False),
            external_inputs=self._external_inputs,
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

    # ----------------------------------------------------------------- action

    def get_action(self) -> RobotAction:
        """Step the session and return the squeeze-clutched per-frame increment.

        ``grip_pos`` / ``grip_quat`` / ``grip_ori`` are **frame-to-frame
        increments**, not totals. The consumer can do ``ee += grip_pos`` and
        ``ee_quat = grip_quat * ee_quat`` directly — no home, no prior delta
        tracking, no integration math.

        On the rising edge of ``squeeze`` (above ``_CLUTCH_ENGAGE_THRESHOLD``),
        the previous-frame pose is seeded to the current raw pose so the
        first engaged frame yields a zero increment (no jump). On subsequent
        engaged frames the increment is ``raw[t] - raw[t-1]`` (position) and
        ``raw[t] * raw[t-1]⁻¹`` (orientation). On release
        (below ``_CLUTCH_RELEASE_THRESHOLD``) and during untracked / invalid
        frames the output is zeroed — the consumer's ``ee`` does not move.

        Axis convention for ``grip_pos`` (operator-friendly; not the base frame):
            x = right (+) / left (-)
            y = front (+) / back (-)
            z = up (+)    / down (-)
        The base-frame increment is mapped via
        ``user = (-base_y, base_x, base_z)``.

        Returns:
            ``{"grip_pos": (3,), "grip_quat": (4,), "grip_ori": (3,) roll/pitch/yaw,
            "squeeze": float, "trigger": float, "a_button": float, "b_button": float}``
        """
        result = self._step()
        controller = result["controller"]

        pos: np.ndarray | None = None
        quat: np.ndarray | None = None
        squeeze_val = 0.0
        trigger_val = 0.0
        a_val = 0.0
        b_val = 0.0
        grip_is_valid = False

        is_tracking = not getattr(controller, "is_none", False)
        if is_tracking:
            try:
                grip_is_valid = bool(controller[ControllerInputIndex.GRIP_IS_VALID])
                pos = np.asarray(controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32)
                quat = np.asarray(controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float32)
                squeeze_val = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
                trigger_val = float(controller[ControllerInputIndex.TRIGGER_VALUE])
                a_val = float(controller[ControllerInputIndex.PRIMARY_CLICK])
                b_val = float(controller[ControllerInputIndex.SECONDARY_CLICK])
            except (IndexError, KeyError, TypeError, ValueError):
                pos = None

        # Frame validity gates both the clutch state machine and the live
        # buttons. A briefly-occluded frame must not advance the state
        # machine (would trigger spurious engage/release) and must not leak
        # a partial read into the output dict.
        frame_ok = (
            pos is not None
            and quat is not None
            and pos.shape == (3,)
            and quat.shape == (4,)
            and grip_is_valid
            and np.all(np.isfinite(pos))
            and np.all(np.isfinite(quat))
            and np.isfinite(squeeze_val)
            and np.isfinite(trigger_val)
        )
        self._is_tracking = frame_ok

        quat_unit: np.ndarray | None = None
        if frame_ok:
            quat_norm = float(np.linalg.norm(quat))
            if quat_norm > 1e-6:
                quat_unit = (quat / quat_norm).astype(np.float32)
            else:
                frame_ok = False  # degenerate quaternion

        if frame_ok:
            assert quat_unit is not None
            # Hysteresis: rising edge above engage threshold seeds prev=raw
            # (so the first engaged frame yields a zero increment); falling
            # edge below release threshold frees the clutch.
            if not self._clutch_engaged and squeeze_val >= self._CLUTCH_ENGAGE_THRESHOLD:
                self._clutch_engaged = True
                self._clutch_prev_pos = pos.copy()
                self._clutch_prev_quat = quat_unit
            elif self._clutch_engaged and squeeze_val < self._CLUTCH_RELEASE_THRESHOLD:
                self._clutch_engaged = False

        if frame_ok and self._clutch_engaged:
            assert self._clutch_prev_pos is not None
            assert self._clutch_prev_quat is not None
            # Per-frame position increment in base frame, then mapped to
            # the operator-friendly axis convention: user = (-base_y,
            # base_x, base_z). So moving the controller right in the real
            # world (which is -base_y) reports as +user_x; forward
            # (+base_x) is +user_y; up (+base_z) is +user_z.
            delta_base = pos - self._clutch_prev_pos
            grip_pos = np.array(
                [-delta_base[1], delta_base[0], delta_base[2]],
                dtype=np.float32,
            )
            # Per-frame orientation increment ``raw * prev⁻¹`` — the
            # rotation that maps prev to raw. Consumer composes via
            # ``ee_quat = grip_quat * ee_quat``.
            rel_rot = Rotation.from_quat(quat_unit) * Rotation.from_quat(self._clutch_prev_quat).inv()
            grip_quat = rel_rot.as_quat().astype(np.float32)
            grip_ori = rel_rot.as_euler("xyz").astype(np.float32)
            # Roll prev forward for next frame.
            self._clutch_prev_pos = pos.copy()
            self._clutch_prev_quat = quat_unit
        else:
            # Not engaged, no prev yet, or untracked: zero increment. When
            # the consumer sees zeros it does not move the EE.
            grip_pos = np.zeros(3, dtype=np.float32)
            grip_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            grip_ori = np.zeros(3, dtype=np.float32)

        return {
            "grip_pos": grip_pos,
            "grip_ori": grip_ori,
            "grip_quat": grip_quat,
            "squeeze": squeeze_val if frame_ok else 0.0,
            "trigger": trigger_val if frame_ok else 0.0,
            "a_button": a_val if frame_ok and np.isfinite(a_val) else 0.0,
            "b_button": b_val if frame_ok and np.isfinite(b_val) else 0.0,
        }


__all__ = ["IsaacTeleop", "IsaacTeleopConfig"]
