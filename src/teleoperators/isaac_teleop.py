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

  * :class:`IsaacTeleopConfig` — a flat :class:`TeleoperatorConfig` registered as
    ``--teleop.type=isaac_teleop`` so draccus can decode the CLI.
  * :class:`IsaacTeleop` — a :class:`Teleoperator` subclass that owns an Isaac
    Teleop ``TeleopSession`` and exposes the raw base-frame grip pose and
    squeeze/trigger values.

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
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


@TeleoperatorConfig.register_subclass("isaac_teleop")
@dataclass
class IsaacTeleopConfig(TeleoperatorConfig): ...


class IsaacTeleop(Teleoperator):
    """Standard Isaac Teleop teleoperator"""

    config_class = IsaacTeleopConfig
    name = "isaac_teleop"

    def __init__(self, config: IsaacTeleopConfig):
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self.session: TeleopSession | None = None
        self.cloudxr_launcher: CloudXRLauncher | None = None
        self.cloudxr_launcher: dict[str, Any] | None = None

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
    def is_connected(self) -> bool:
        return self.session is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Tracking devices self-calibrate.

    def calibrate(self) -> None: ...

    def configure(self) -> None: ...

    def send_feedback(self, feedback: dict[str, Any]) -> None: ...

    def connect(self, calibrate: bool = True) -> None:
        if self.session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        if calibrate:
            self.calibrate()

        self.ensure_cloudxr_runtime()
        try:
            # build pipeline
            controllers = ControllersSource(name="controllers")
            xform = ValueInput("base_T_anchor", TransformMatrix())
            transformed = controllers.transformed(xform.output("value"))
            ctrl = transformed.output("controller_right")
            pipeline = OutputCombiner({"controller": ctrl})
            session_config = TeleopSessionConfig(app_name=self.__class__.__name__, pipeline=pipeline)
            self.session = TeleopSession(session_config)
            self.session.__enter__()
            # self.cloudxr_launcher = self._buildcloudxr_launcher()
        except Exception:
            self.session = None
            self.cloudxr_launcher = None
            try:
                self._stop_cloudxr_runtime()
            except Exception:
                logger.exception("Failed to stop CloudXR runtime during connect() rollback")
            raise
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def disconnect(self) -> None:
        try:
            if self.session is not None:
                # Null the handle BEFORE __exit__: even a failed session teardown must
                # not wedge the device as is_connected.
                session = self.session
                self.session = None
                self.cloudxr_launcher = None
                session.__exit__(None, None, None)
                logger.info("Isaac Teleop session ended")
        finally:
            self._stop_cloudxr_runtime()

    # -------------------------------------------------------- CloudXR runtime

    def ensure_cloudxr_runtime(self) -> None:
        """Auto-launch the CloudXR runtime once, unless opted out."""
        if self.cloudxr_launcher is not None:
            return

        if os.environ.get("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "").strip() == "1":
            logger.info("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1 set; skipping CloudXR auto-launch.")
            return

        logger.info("Launching CloudXR runtime (first run may prompt for EULA and take ~30s)...")
        self.cloudxr_launcher = CloudXRLauncher(install_dir=str(Path.home() / ".cloudxr"), accept_eula=False)

    def _stop_cloudxr_runtime(self) -> None:
        if self.cloudxr_launcher is None:
            return
        try:
            self.cloudxr_launcher.stop()
        except RuntimeError:
            logger.warning("CloudXR runtime could not be terminated; handle retained for atexit cleanup")
        else:
            self.cloudxr_launcher = None
            logger.info("CloudXR runtime stopped")

    # ------------------------------------------------------------ pipeline

    def _buildcloudxr_launcher(self) -> dict[str, Any]:
        """Materialize the constant ``base_T_anchor`` external input (once, in connect)."""
        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        return {"base_T_anchor": {"value": tg}}

    def _step(self) -> Any:
        if self.session is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self.session.step(
            execution_events=ExecutionEvents(execution_state=ExecutionState.RUNNING, reset=False),
            external_inputs=self.cloudxr_launcher,
        )
        info = self.session.last_step_info
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


if __name__ == "__main__":
    config = IsaacTeleopConfig()
    teleop = IsaacTeleop(config)
    teleop.connect()
    while True:
        action = teleop.get_action()
        print(action)
