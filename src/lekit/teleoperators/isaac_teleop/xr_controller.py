"""Quest/OpenXR controller exposed as a LeRobot teleoperator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.types import RobotAction

from .config import IsaacTeleopConfig
from .protocol import CONTROLLER_SIDES, action_features
from .relative_pose import RelativePoseClutch, default_operator_frame, operator_frame_from_head_quaternion
from .session import IsaacTeleopSession, _isaacteleop_available

_POSE_NAMES = ("grip", "aim")

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HeadSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner
    from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex, HeadPoseIndex
else:
    ControllersSource = Any
    HeadSource = Any
    OutputCombiner = Any
    ControllerInputIndex = Any
    HeadPoseIndex = Any


class IsaacXRController(IsaacTeleopSession):
    """Return both Quest controllers in a single flat LeRobot action.

    ``{side}.translation`` and ``{side}.aim_translation`` use metres in
    ``(right, forward, up)`` order. Their corresponding rotations are ``xyzw``
    quaternions relative to each pose at squeeze engagement. These values are
    input-device data, not robot or TCP commands.
    """

    config_class = IsaacTeleopConfig
    name = "isaac_xr_controller"

    def __init__(self, config: IsaacTeleopConfig):
        super().__init__(config)
        self._clutches = {
            side: {
                pose_name: RelativePoseClutch(
                    engage_threshold=config.squeeze_engage_threshold,
                    release_threshold=config.squeeze_release_threshold,
                )
                for pose_name in _POSE_NAMES
            }
            for side in CONTROLLER_SIDES
        }
        self._is_tracking = dict.fromkeys(CONTROLLER_SIDES, False)

    @property
    def action_features(self) -> dict:
        return action_features()

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_tracking(self) -> bool:
        """Whether both controllers currently have valid grip tracking."""

        return all(self._is_tracking.values())

    def disconnect(self) -> None:
        for hand_clutches in self._clutches.values():
            for clutch in hand_clutches.values():
                clutch.reset()
        self._is_tracking = dict.fromkeys(CONTROLLER_SIDES, False)
        super().disconnect()

    def _build_pipeline(self) -> OutputCombiner:
        controllers = ControllersSource(name="controllers")
        head = HeadSource(name="head").output("head")
        return OutputCombiner(
            {
                "controller_left": controllers.output("controller_left"),
                "controller_right": controllers.output("controller_right"),
                "head": head,
            }
        )

    def get_action(self) -> RobotAction:
        result = self._step()
        head_quaternion = self._read_head_quaternion(result.get("head"))
        operator_to_anchor = (
            operator_frame_from_head_quaternion(head_quaternion)
            if self.config.use_head_yaw
            else default_operator_frame()
        )
        action: RobotAction = {}
        for side in CONTROLLER_SIDES:
            action.update(
                self._hand_action(
                    side,
                    self._read_controller(result[f"controller_{side}"]),
                    operator_to_anchor,
                )
            )
        return action

    def _hand_action(
        self,
        side: str,
        raw: dict[str, Any],
        operator_to_anchor: np.ndarray | None,
    ) -> RobotAction:
        grip_pose = self._clutches[side]["grip"].update(
            position=raw["position"],
            quaternion=raw["quaternion"],
            squeeze=raw["squeeze"],
            operator_to_anchor=operator_to_anchor,
            tracked=raw["tracked"],
        )
        aim_pose = self._clutches[side]["aim"].update(
            position=raw["aim_position"],
            quaternion=raw["aim_quaternion"],
            squeeze=raw["squeeze"],
            operator_to_anchor=operator_to_anchor,
            tracked=bool(raw["aim_tracked"] and grip_pose.engaged),
        )
        self._is_tracking[side] = bool(raw["tracked"])
        return {
            f"{side}.translation": grip_pose.translation,
            f"{side}.rotation": grip_pose.rotation,
            f"{side}.aim_translation": aim_pose.translation,
            f"{side}.aim_rotation": aim_pose.rotation,
            f"{side}.squeeze": raw["squeeze"],
            f"{side}.trigger": raw["trigger"],
            f"{side}.thumbstick": np.array([raw["thumbstick_x"], raw["thumbstick_y"]], dtype=np.float32),
            f"{side}.thumbstick_click": raw["thumbstick_click"],
            f"{side}.primary_button": raw["primary_button"],
            f"{side}.secondary_button": raw["secondary_button"],
            f"{side}.menu_button": raw["menu_button"],
            f"{side}.is_tracking": self._is_tracking[side],
            f"{side}.is_aim_tracking": bool(raw["aim_tracked"]),
            f"{side}.is_engaged": grip_pose.engaged,
        }

    @staticmethod
    def _read_controller(controller: Any) -> dict[str, Any]:
        empty = {
            "position": np.zeros(3, dtype=np.float32),
            "quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "aim_position": np.zeros(3, dtype=np.float32),
            "aim_quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "squeeze": 0.0,
            "trigger": 0.0,
            "thumbstick_x": 0.0,
            "thumbstick_y": 0.0,
            "thumbstick_click": 0.0,
            "primary_button": 0.0,
            "secondary_button": 0.0,
            "menu_button": 0.0,
            "tracked": False,
            "aim_tracked": False,
        }
        if getattr(controller, "is_none", True):
            return empty
        values = dict(empty)
        position, quaternion, tracked = IsaacXRController._read_pose(
            controller,
            ControllerInputIndex.GRIP_POSITION,
            ControllerInputIndex.GRIP_ORIENTATION,
            ControllerInputIndex.GRIP_IS_VALID,
        )
        aim_position, aim_quaternion, aim_tracked = IsaacXRController._read_pose(
            controller,
            ControllerInputIndex.AIM_POSITION,
            ControllerInputIndex.AIM_ORIENTATION,
            ControllerInputIndex.AIM_IS_VALID,
        )
        values.update(
            position=position,
            quaternion=quaternion,
            tracked=tracked,
            aim_position=aim_position,
            aim_quaternion=aim_quaternion,
            aim_tracked=aim_tracked,
        )
        for key, index, limits in (
            ("squeeze", ControllerInputIndex.SQUEEZE_VALUE, (0.0, 1.0)),
            ("trigger", ControllerInputIndex.TRIGGER_VALUE, (0.0, 1.0)),
            ("primary_button", ControllerInputIndex.PRIMARY_CLICK, (0.0, 1.0)),
            ("secondary_button", ControllerInputIndex.SECONDARY_CLICK, (0.0, 1.0)),
            ("menu_button", ControllerInputIndex.MENU_CLICK, (0.0, 1.0)),
            ("thumbstick_x", ControllerInputIndex.THUMBSTICK_X, (-1.0, 1.0)),
            ("thumbstick_y", ControllerInputIndex.THUMBSTICK_Y, (-1.0, 1.0)),
            ("thumbstick_click", ControllerInputIndex.THUMBSTICK_CLICK, (0.0, 1.0)),
        ):
            try:
                value = float(controller[index])
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values[key] = float(np.clip(value, *limits))
        return values

    @staticmethod
    def _read_pose(
        controller: Any,
        position_index: Any,
        quaternion_index: Any,
        valid_index: Any,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        position = np.zeros(3, dtype=np.float32)
        quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        try:
            candidate_position = np.asarray(controller[position_index], dtype=np.float32)
            candidate_quaternion = np.asarray(controller[quaternion_index], dtype=np.float32)
            valid = bool(controller[valid_index])
        except (IndexError, KeyError, TypeError, ValueError):
            return position, quaternion, False
        quaternion_norm = float(np.linalg.norm(candidate_quaternion))
        if not (
            valid
            and candidate_position.shape == (3,)
            and candidate_quaternion.shape == (4,)
            and np.all(np.isfinite(candidate_position))
            and np.all(np.isfinite(candidate_quaternion))
            and quaternion_norm > 1e-6
        ):
            return position, quaternion, False
        return candidate_position, candidate_quaternion / quaternion_norm, True

    @staticmethod
    def _read_head_quaternion(head: Any) -> np.ndarray | None:
        if head is None or getattr(head, "is_none", True):
            return None
        try:
            valid = bool(head[HeadPoseIndex.IS_VALID])
            quaternion = np.asarray(head[HeadPoseIndex.ORIENTATION], dtype=np.float32)
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        norm = float(np.linalg.norm(quaternion))
        if not (valid and quaternion.shape == (4,) and np.all(np.isfinite(quaternion)) and norm > 1e-6):
            return None
        return quaternion / norm


IsaacTeleop = IsaacXRController

__all__ = ["CONTROLLER_SIDES", "IsaacTeleop", "IsaacXRController"]
