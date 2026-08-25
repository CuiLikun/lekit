"""Contract tests for the robot-independent XR clutch state machine."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lekit.teleoperators.isaac_teleop.config import IsaacTeleopConfig
from lekit.teleoperators.isaac_teleop.relative_pose import (
    RelativePoseClutch,
    default_operator_frame,
    operator_frame_from_head_quaternion,
)
from lekit.teleoperators.isaac_teleop.xr_controller import IsaacXRController


def _clutch() -> RelativePoseClutch:
    return RelativePoseClutch(engage_threshold=0.5, release_threshold=0.3)


def _update(
    clutch: RelativePoseClutch,
    position: list[float],
    *,
    squeeze: float,
    quaternion: list[float] | None = None,
    frame: np.ndarray | None = None,
    tracked: bool = True,
):
    return clutch.update(
        position=np.asarray(position),
        quaternion=np.asarray(quaternion or [0.0, 0.0, 0.0, 1.0]),
        squeeze=squeeze,
        operator_to_anchor=default_operator_frame() if frame is None else frame,
        tracked=tracked,
    )


def test_relative_translation_is_cumulative_in_operator_axes() -> None:
    clutch = _clutch()
    neutral = _update(clutch, [1.0, 2.0, 3.0], squeeze=0.0)
    assert not neutral.engaged
    np.testing.assert_array_equal(neutral.translation, [0.0, 0.0, 0.0])

    engaged = _update(clutch, [1.0, 2.0, 3.0], squeeze=0.6)
    assert engaged.engaged
    np.testing.assert_array_equal(engaged.translation, [0.0, 0.0, 0.0])

    # OpenXR +X is right, -Z is forward, and +Y is up.
    np.testing.assert_allclose(_update(clutch, [1.2, 2.0, 2.9], squeeze=0.6).translation, [0.2, 0.1, 0.0])
    np.testing.assert_allclose(_update(clutch, [0.8, 2.05, 3.0], squeeze=0.6).translation, [-0.2, 0.0, 0.05])


def test_release_and_tracking_loss_return_neutral_and_require_reengage() -> None:
    clutch = _clutch()
    _update(clutch, [0.0, 0.0, 0.0], squeeze=0.6)
    released = _update(clutch, [0.0, 0.0, -0.1], squeeze=0.2)
    assert not released.engaged
    np.testing.assert_array_equal(released.translation, [0.0, 0.0, 0.0])

    lost = _update(clutch, [0.0, 0.0, -0.1], squeeze=0.8, tracked=False)
    assert not lost.engaged
    assert not clutch.engaged
    reengaged = _update(clutch, [0.0, 0.0, -0.1], squeeze=0.8)
    np.testing.assert_array_equal(reengaged.translation, [0.0, 0.0, 0.0])


def test_relative_rotation_is_expressed_in_operator_frame() -> None:
    clutch = _clutch()
    _update(clutch, [0.0, 0.0, 0.0], squeeze=0.6)
    # A +90 degree OpenXR-Y rotation is a +90 degree operator-Z rotation.
    quarter_turn = np.sqrt(0.5)
    relative = _update(
        clutch, [0.0, 0.0, 0.0], squeeze=0.6, quaternion=[0.0, quarter_turn, 0.0, quarter_turn]
    )
    np.testing.assert_allclose(relative.rotation, [0.0, 0.0, quarter_turn, quarter_turn], atol=1e-6)


def test_head_frame_is_latched_by_the_caller_on_engage() -> None:
    identity = operator_frame_from_head_quaternion(np.array([0.0, 0.0, 0.0, 1.0]))
    assert identity is not None
    np.testing.assert_allclose(identity, default_operator_frame())
    assert operator_frame_from_head_quaternion(np.array([0.0, 0.0, 0.0, 0.0])) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"squeeze_engage_threshold": 0.2, "squeeze_release_threshold": 0.3}, "release"),
        ({"squeeze_engage_threshold": float("nan")}, "engage"),
    ],
)
def test_config_rejects_invalid_clutch_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IsaacTeleopConfig(**kwargs)


class _Frame(dict):
    is_none = False


def test_config_has_no_single_hand_selection() -> None:
    assert "hand_side" not in IsaacTeleopConfig.__dataclass_fields__


def _controller_frame(*, position: list[float], squeeze: float, primary: float) -> _Frame:
    return _Frame(
        position=position,
        quaternion=[0.0, 0.0, 0.0, 1.0],
        valid=True,
        aim_position=np.add(position, [0.0, 0.0, -0.1]).tolist(),
        aim_quaternion=[0.0, 0.0, 0.0, 1.0],
        aim_valid=True,
        squeeze=squeeze,
        trigger=0.25,
        primary=primary,
        secondary=0.0,
        menu=1.0 if primary == 0.0 else 0.0,
        stick_x=0.5,
        stick_y=-0.25,
        stick_click=1.0,
    )


def _bare_controller() -> IsaacXRController:
    controller = object.__new__(IsaacXRController)
    controller.config = IsaacTeleopConfig(use_head_yaw=False)
    controller._clutches = {
        side: {
            pose: RelativePoseClutch(engage_threshold=0.5, release_threshold=0.3) for pose in ("grip", "aim")
        }
        for side in ("left", "right")
    }
    controller._is_tracking = {"left": False, "right": False}
    return controller


def test_controller_get_action_exposes_both_controllers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise dual input decoding and the public action shape without XR hardware."""

    from lekit.teleoperators.isaac_teleop import xr_controller

    controller_index = SimpleNamespace(
        GRIP_POSITION="position",
        GRIP_ORIENTATION="quaternion",
        AIM_POSITION="aim_position",
        AIM_ORIENTATION="aim_quaternion",
        SQUEEZE_VALUE="squeeze",
        TRIGGER_VALUE="trigger",
        GRIP_IS_VALID="valid",
        AIM_IS_VALID="aim_valid",
        PRIMARY_CLICK="primary",
        SECONDARY_CLICK="secondary",
        MENU_CLICK="menu",
        THUMBSTICK_X="stick_x",
        THUMBSTICK_Y="stick_y",
        THUMBSTICK_CLICK="stick_click",
    )
    monkeypatch.setattr(xr_controller, "ControllerInputIndex", controller_index)

    frames = iter(
        [
            {
                "controller_left": _controller_frame(position=[-1.0, 2.0, 3.0], squeeze=0.6, primary=0.0),
                "controller_right": _controller_frame(position=[1.0, 2.0, 3.0], squeeze=0.6, primary=1.0),
                "head": None,
            },
            {
                "controller_left": _controller_frame(position=[-0.9, 2.0, 2.8], squeeze=0.6, primary=0.0),
                "controller_right": _controller_frame(position=[0.8, 2.05, 2.9], squeeze=0.6, primary=1.0),
                "head": None,
            },
        ]
    )
    controller = _bare_controller()
    controller._step = lambda: next(frames)

    first = controller.get_action()
    second = controller.get_action()

    expected_fields = {
        "translation",
        "rotation",
        "aim_translation",
        "aim_rotation",
        "squeeze",
        "trigger",
        "thumbstick",
        "thumbstick_click",
        "primary_button",
        "secondary_button",
        "menu_button",
        "is_tracking",
        "is_aim_tracking",
        "is_engaged",
    }
    assert set(first) == {f"{side}.{field}" for side in ("left", "right") for field in expected_fields}
    assert first["left.is_tracking"] and first["right.is_tracking"]
    assert first["left.is_engaged"] and first["right.is_engaged"]
    np.testing.assert_array_equal(first["left.translation"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(second["left.translation"], [0.1, 0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(second["right.translation"], [-0.2, 0.1, 0.05], atol=1e-6)
    np.testing.assert_allclose(second["right.aim_translation"], [-0.2, 0.1, 0.05], atol=1e-6)
    np.testing.assert_allclose(second["left.thumbstick"], [0.5, -0.25])
    assert second["left.menu_button"] == 1.0
    assert second["right.primary_button"] == 1.0
    assert second["right.thumbstick_click"] == 1.0


def test_action_features_match_dual_controller_action_contract() -> None:
    controller = _bare_controller()

    assert set(controller.action_features) == {
        f"{side}.{field}"
        for side in ("left", "right")
        for field in (
            "translation",
            "rotation",
            "aim_translation",
            "aim_rotation",
            "squeeze",
            "trigger",
            "thumbstick",
            "thumbstick_click",
            "primary_button",
            "secondary_button",
            "menu_button",
            "is_tracking",
            "is_aim_tracking",
            "is_engaged",
        )
    }


def test_invalid_pose_does_not_erase_controller_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from lekit.teleoperators.isaac_teleop import xr_controller

    controller_index = SimpleNamespace(
        GRIP_POSITION="position",
        GRIP_ORIENTATION="quaternion",
        GRIP_IS_VALID="valid",
        AIM_POSITION="aim_position",
        AIM_ORIENTATION="aim_quaternion",
        AIM_IS_VALID="aim_valid",
        SQUEEZE_VALUE="squeeze",
        TRIGGER_VALUE="trigger",
        PRIMARY_CLICK="primary",
        SECONDARY_CLICK="secondary",
        MENU_CLICK="menu",
        THUMBSTICK_X="stick_x",
        THUMBSTICK_Y="stick_y",
        THUMBSTICK_CLICK="stick_click",
    )
    monkeypatch.setattr(xr_controller, "ControllerInputIndex", controller_index)
    frame = _controller_frame(position=[1.0, 2.0, 3.0], squeeze=0.75, primary=1.0)
    frame["valid"] = False
    frame["aim_valid"] = False

    raw = IsaacXRController._read_controller(frame)

    assert not raw["tracked"] and not raw["aim_tracked"]
    np.testing.assert_array_equal(raw["position"], [0.0, 0.0, 0.0])
    assert raw["squeeze"] == 0.75
    assert raw["trigger"] == 0.25
    assert raw["primary_button"] == 1.0
    assert raw["thumbstick_x"] == 0.5
    assert raw["thumbstick_click"] == 1.0


def test_controller_pipeline_reads_left_and_right(monkeypatch: pytest.MonkeyPatch) -> None:
    from lekit.teleoperators.isaac_teleop import xr_controller

    class _Source:
        def __init__(self, name: str):
            self.name = name

        def output(self, name: str) -> str:
            return f"{self.name}.{name}"

    monkeypatch.setattr(xr_controller, "ControllersSource", _Source)
    monkeypatch.setattr(xr_controller, "HeadSource", _Source)
    monkeypatch.setattr(xr_controller, "OutputCombiner", lambda outputs: outputs)

    controller = _bare_controller()

    assert controller._build_pipeline() == {
        "controller_left": "controllers.controller_left",
        "controller_right": "controllers.controller_right",
        "head": "head.head",
    }
