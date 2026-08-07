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

"""Standard Isaac Teleop teleoperator (gamepad-shaped).

Defines:

  * :class:`IsaacTeleopConfig` — a flat :class:`TeleoperatorConfig` registered as
    ``--teleop.type=isaac_teleop`` so draccus can decode the CLI.
  * :class:`IsaacTeleop` — a :class:`Teleoperator` subclass that owns an Isaac
    Teleop ``TeleopSession`` and exposes the raw base-frame grip pose and
    squeeze/trigger values.

Mirrors the structure of
``.venv/lib/python3.12/site-packages/lerobot/teleoperators/gamepad``: no
shared base / custom choice registry, no retargeters — just a
``Teleoperator`` and its config.

For the recorder-side rebasing (operator yaw, lock-pose, servo profile),
use :mod:`examples.isaac_teleop_to_jaka.xr` and its
``make_xr_device`` bundle instead.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.import_utils import is_package_available

logger = logging.getLogger(__name__)

# isaacteleop is an optional NVIDIA dep. Guard the import so this module loads
# without it (and construction fails fast with install instructions instead of
# at first use).
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

# Identity rebase between the CloudXR anchor frame and the target base frame.
# Override via ``base_T_anchor`` when the consumer wants a rotated frame.
DEFAULT_BASE_T_ANCHOR = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# Source-node input name fed each step via TeleopSession.step.
_BASE_T_ANCHOR_INPUT = "base_T_anchor"


def _require_isaacteleop() -> None:
    if not _isaacteleop_available:
        raise ImportError(
            "The 'isaacteleop' package is required for IsaacTeleop but is not "
            "installed. See examples/isaac_teleop_to_jaka/README.md for install instructions."
        )


@TeleoperatorConfig.register_subclass("isaac_teleop")
@dataclass
class IsaacTeleopConfig(TeleoperatorConfig):
    """Config for the standard :class:`IsaacTeleop` teleoperator.

    Registered with ``--teleop.type=isaac_teleop`` so draccus can decode the CLI.
    """

    hand_side: str = "right"
    """Which controller hand to drive: ``"left"`` or ``"right"``."""

    app_name: str = "LeTeleop"
    """Application name for the OpenXR / Isaac Teleop session."""

    auto_launch_cloudxr: bool = True
    """Auto-launch the CloudXR runtime on :meth:`IsaacTeleop.connect`. Set
    ``False`` (or export ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1``) when CloudXR
    runs externally."""

    cloudxr_env_file: str | None = None
    """Optional CloudXR device-profile ``.env`` passed to ``CloudXRLauncher``."""

    base_T_anchor: list[list[float]] | None = None  # noqa: N815
    """Static 4x4 transform rebasing the OpenXR anchor frame into the target
    base frame. Defaults to identity (no rebase)."""

    def __post_init__(self):
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {self.hand_side!r}")
        if self.base_T_anchor is None:
            self.base_T_anchor = [row[:] for row in DEFAULT_BASE_T_ANCHOR]


class IsaacTeleop(Teleoperator):
    """Standard Isaac Teleop teleoperator (gamepad-shaped).

    Owns a :class:`TeleopSession`, auto-launches CloudXR on first connect, and
    returns the raw grip pose + squeeze/trigger of the requested hand on each
    :meth:`get_action`.
    """

    config_class = IsaacTeleopConfig
    name = "isaac_teleop"

    def __init__(self, config: IsaacTeleopConfig):
        _require_isaacteleop()
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self._session: TeleopSession | None = None
        self._cloudxr_launcher: CloudXRLauncher | None = None
        self._external_inputs: dict[str, Any] | None = None

    # ------------------------------------------------------------------ schema

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

    # ------------------------------------------------------------ lifecycle

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Tracking devices self-calibrate.

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not yet implemented.

    def connect(self, calibrate: bool = True) -> None:
        _require_isaacteleop()
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        self._ensure_cloudxr_runtime()
        try:
            pipeline = self._build_pipeline()
            session_config = TeleopSessionConfig(app_name=self.config.app_name, pipeline=pipeline)
            self._session = TeleopSession(session_config)
            self._session.__enter__()
            self._external_inputs = self._build_external_inputs()
        except Exception:
            self._session = None
            self._external_inputs = None
            try:
                self._stop_cloudxr_runtime()
            except Exception:
                logger.exception("Failed to stop CloudXR runtime during connect() rollback")
            raise
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def disconnect(self) -> None:
        try:
            if self._session is not None:
                # Null the handle BEFORE __exit__: even a failed session teardown must
                # not wedge the device as is_connected.
                session = self._session
                self._session = None
                self._external_inputs = None
                session.__exit__(None, None, None)
                logger.info("Isaac Teleop session ended")
        finally:
            self._stop_cloudxr_runtime()

    # -------------------------------------------------------- CloudXR runtime

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

    # ------------------------------------------------------------ pipeline

    def _build_pipeline(self) -> OutputCombiner:
        controller_key = f"controller_{self.config.hand_side}"
        controllers = ControllersSource(name="controllers")
        xform = ValueInput(_BASE_T_ANCHOR_INPUT, TransformMatrix())
        ctrl = controllers.transformed(xform.output("value")).output(controller_key)
        return OutputCombiner({"controller": ctrl})

    def _build_external_inputs(self) -> dict[str, Any]:
        """Materialize the constant ``base_T_anchor`` external input (once, in connect)."""
        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        return {_BASE_T_ANCHOR_INPUT: {"value": tg}}

    def _step(self) -> Any:
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

        On any read/validation failure the defaults stand untouched (identity
        quat, zero position, zero squeeze/trigger) — a partially-populated
        frame must not mix live values with the safe defaults.
        """
        result = self._step()
        controller = result["controller"]

        grip_pos = np.zeros(3, dtype=np.float32)
        grip_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        squeeze = 0.0
        trigger = 0.0
        if not getattr(controller, "is_none", False):
            try:
                grip_is_valid = bool(controller[ControllerInputIndex.GRIP_IS_VALID])
                pos = np.asarray(controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32)
                quat = np.asarray(controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float32)
                squeeze_val = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
                trigger_val = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            except (IndexError, KeyError, TypeError, ValueError):
                pass
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
