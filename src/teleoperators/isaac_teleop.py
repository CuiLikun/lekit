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

    def __init__(self, config: IsaacTeleopConfig):
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self._session: TeleopSession | None = None
        self._cloudxr_launcher: CloudXRLauncher | None = None
        self._external_inputs: dict[str, Any] | None = None
        self._is_tracking = False

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
        """Step the session and return the raw base-frame grip pose + squeeze/trigger.

        Returns identity quat / zero position / zero squeeze+trigger when the
        controller is not tracked or the read fails — a partially-populated
        frame must never mix live values with safe defaults.
        """
        result = self._step()
        controller = result["controller"]

        grip_pos = np.zeros(3, dtype=np.float32)
        grip_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        squeeze = 0.0
        trigger = 0.0

        self._is_tracking = not getattr(controller, "is_none", False)
        if not self._is_tracking:
            return {
                "grip_pos": grip_pos,
                "grip_quat": grip_quat,
                "squeeze": squeeze,
                "trigger": trigger,
            }

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
                grip_is_valid
                and pos.shape == (3,)
                and quat.shape == (4,)
                and np.all(np.isfinite(pos))
                and np.all(np.isfinite(quat))
                and np.isfinite(squeeze_val)
                and np.isfinite(trigger_val)
                and quat_norm > 1e-6
            ):
                grip_pos = pos
                grip_quat = quat / quat_norm
                squeeze = squeeze_val
                trigger = trigger_val

        return {
            "grip_pos": grip_pos,
            "grip_quat": grip_quat,
            "squeeze": squeeze,
            "trigger": trigger,
        }


__all__ = ["IsaacTeleop", "IsaacTeleopConfig"]


if __name__ == "__main__":
    config = IsaacTeleopConfig()
    teleop = IsaacTeleop(config)
    teleop.connect()
    while True:
        action = teleop.get_action()
        print(action)
