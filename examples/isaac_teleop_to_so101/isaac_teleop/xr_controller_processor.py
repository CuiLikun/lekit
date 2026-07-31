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

"""Processor step that maps XR controller actions to robot EE targets.

Analogous to ``MapPhoneActionToRobotAction``, this bridges the clutch-rebased EE pose to
the IK pipeline's input contract (``EEBoundsAndSafety`` -> ``InverseKinematicsEEToJoints``).
Pure (no ``isaacteleop``), so it is unit-testable without the XR runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction
from lerobot.utils.rotation import Rotation

from .base import _GRIPPER_MOTOR_SCALE


def _matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 3x3 rotation matrix into XYZ-Euler ``(roll, pitch, yaw)`` in radians.

    Mirrors the AgileX SDK convention (XYZ intrinsic Euler, equivalent to
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``). Implemented inline (rather than
    via scipy) to avoid pulling in a new top-level dependency for the
    single use site.
    """
    r00, r01 = matrix[0, 0], matrix[0, 1]
    r11 = matrix[1, 1]
    r20, r21, r22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    # Pitch: clamp to the open interval (-pi/2, pi/2) so asin stays defined;
    # gimbal lock clamps to ±pi/2 and falls back to atan2 for roll/yaw.
    pitch = float(np.arcsin(max(-1.0, min(1.0, -r20))))
    if abs(r20) < 1.0 - 1e-9:
        roll = float(np.arctan2(r21, r22))
        yaw = float(np.arctan2(matrix[1, 0], r00))
    else:
        # Gimbal lock: pick roll=0 and back out yaw from r01/r11.
        roll = 0.0
        yaw = float(np.arctan2(-r01, r11))
    return roll, pitch, yaw


@ProcessorStepRegistry.register("map_xr_controller_action_to_robot_action")
@dataclass
class MapXRControllerActionToRobotAction(RobotActionProcessorStep):
    """Maps an absolute base-frame EE pose + gripper closedness to the IK input contract.

    Pure, stateless rename (the owning loop's clutch already produced the absolute base-frame
    target). Each frame it writes:

    - ``ee.x/y/z`` = ``ee_pose[:3]`` (position [m]);
    - ``ee.wx/wy/wz`` = rotvec of ``ee_pose[3:7]`` (orientation; the IK tracks it softly at a
      small ``orientation_weight`` on the 5-DOF SO-101);
    - ``ee.gripper_pos`` = ``(1 - closedness) * _GRIPPER_MOTOR_SCALE`` (jaw target [0, 100],
      RANGE_0_100 where 100 = open, so closedness is inverted).

    Input keys: ``ee_pose`` ``(7,)`` ``[x,y,z,qx,qy,qz,qw]``, ``closedness`` float in [0, 1].
    """

    def action(self, action: RobotAction) -> RobotAction:
        ee_pose = action.pop("ee_pose")
        closedness = float(action.pop("closedness"))

        action["ee.x"] = float(ee_pose[0])
        action["ee.y"] = float(ee_pose[1])
        action["ee.z"] = float(ee_pose[2])
        # Orientation target as a rotvec (quat [qx,qy,qz,qw] -> axis-angle); the IK
        # consumes ee.w* as a rotvec and tracks it with orientation_weight.
        rotvec = Rotation.from_quat(ee_pose[3:7]).as_rotvec()
        action["ee.wx"] = float(rotvec[0])
        action["ee.wy"] = float(rotvec[1])
        action["ee.wz"] = float(rotvec[2])
        # Inverted: closedness c=1 (closed) -> 0, c=0 (open) -> 100 (SO-101 calibration).
        action["ee.gripper_pos"] = (1.0 - closedness) * _GRIPPER_MOTOR_SCALE
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["ee_pose", "closedness"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.wx",
            "ee.wy",
            "ee.wz",
            "ee.gripper_pos",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features


@ProcessorStepRegistry.register("map_ee_pose_rotvec_to_agx_arm_rpy")
@dataclass
class MapEEPoseRotVecToAgxArmRPY(RobotActionProcessorStep):
    """Convert IK-pipeline rotvec outputs into the AgxArm ``move_p`` contract.

    Sits after ``EEBoundsAndSafety`` when the follower is an
    :class:`AgxArm` in ``control_mode="ee_pose"``. Drops the IK step
    entirely (AgxArm's firmware handles IK internally via ``move_p``).

    Per-frame it rewrites:

    - ``ee.wx`` / ``ee.wy`` / ``ee.wz`` (rotvec, radians) → ``ee.roll`` /
      ``ee.pitch`` / ``ee.yaw`` (XYZ-Euler radians; the AgileX SDK convention).
    - ``ee.gripper_pos`` ∈ [0, 100] → ``gripper.pos`` ∈ [0, 1]
      (``100 = open`` → ``1.0``; the AgxArm driver scales to metres in
      :meth:`AgxArm.send_action`).

    Pure (stateless, no ``isaacteleop``).
    """

    def action(self, action: RobotAction) -> RobotAction:
        for key in ("ee.wx", "ee.wy", "ee.wz"):
            if key not in action:
                raise ValueError(
                    f"MapEEPoseRotVecToAgxArmRPY.action: missing {key!r} (expected rotvec "
                    "outputs from MapXRControllerActionToRobotAction + EEBoundsAndSafety)."
                )
        rotvec = np.array(
            [float(action.pop("ee.wx")), float(action.pop("ee.wy")), float(action.pop("ee.wz"))],
            dtype=float,
        )
        # rotvec → matrix → XYZ-Euler (AgileX SDK's move_p expects RPY).
        matrix = Rotation.from_rotvec(rotvec).as_matrix()
        action["ee.roll"], action["ee.pitch"], action["ee.yaw"] = _matrix_to_rpy(matrix)

        if "ee.gripper_pos" in action:
            action["gripper.pos"] = float(
                np.clip(action.pop("ee.gripper_pos") / _GRIPPER_MOTOR_SCALE, 0.0, 1.0)
            )
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for feat in ("ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"):
            action_features.pop(feat, None)
        for feat in ("ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"):
            action_features[feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features
