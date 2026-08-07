import importlib
import math

import numpy as np
import pytest

from robots.jaka_robot import jaka_robot as jaka_driver


def _xr_module():
    return importlib.import_module("examples.isaac_teleop_to_jaka.xr")


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


def test_default_xr_base_transform_matches_jaka_operator_axes():
    """Right/forward/up controller motion must map to the matching robot axes."""
    xr = _xr_module()
    rotation = np.asarray(xr.XRControllerConfig().base_T_anchor, dtype=float)[:3, :3]
    yaw = np.deg2rad(xr.DEFAULT_OPERATOR_YAW_DEG)
    operator_right = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    operator_forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])

    assert rotation @ operator_right == pytest.approx([0.0, -1.0, 0.0])
    assert rotation @ operator_forward == pytest.approx([1.0, 0.0, 0.0])
    assert rotation @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])
    assert rotation.T @ rotation == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_lock_pose_holds_measured_orientation_while_translation_follows_controller(monkeypatch):
    xr = _xr_module()
    measured_pose = {
        "ee.x": 0.4,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.roll": 0.1,
        "ee.pitch": -0.2,
        "ee.yaw": 0.3,
    }

    class FakeTeleop:
        def __init__(self, _config):
            self._actions = iter(
                [
                    {
                        "grip_pos": np.array([0.0, 0.0, 0.0]),
                        "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                        "squeeze": 1.0,
                        "trigger": 0.0,
                    },
                    {
                        "grip_pos": np.array([0.05, 0.0, 0.0]),
                        "grip_quat": np.array([0.0, 0.0, math.sin(0.5), math.cos(0.5)]),
                        "squeeze": 1.0,
                        "trigger": 0.0,
                    },
                ]
            )
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def get_action(self):
            return next(self._actions)

        def disconnect(self):
            self.is_connected = False

    class FakeRobot:
        name = "jaka_robot"
        is_connected = True

        def get_eef_pose(self):
            return tuple(
                measured_pose[key]
                for key in ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            )

        def servo_enable(self, _enabled, *, representation="joints"):
            pass

    monkeypatch.setattr(xr, "XRController", FakeTeleop)
    monkeypatch.setattr(xr, "_wait_for_xr_controller", lambda _teleop: None)
    config = xr.XRControllerConfig(lock_pose=True)
    bundle = xr.make_xr_device(FakeRobot(), config)
    bundle["startup"]()

    first = bundle["compute"](measured_pose)
    second = bundle["compute"](measured_pose)

    assert first is not None and second is not None
    assert any(
        second[f"ee.{axis}"] != pytest.approx(first[f"ee.{axis}"])
        for axis in ("x", "y", "z")
    )
    assert [second[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw")] == pytest.approx(
        [measured_pose[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw")]
    )


def test_stationary_xr_position_noise_does_not_change_cartesian_target(monkeypatch):
    xr = _xr_module()
    measured_pose = {
        "ee.x": 0.4,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.roll": 0.1,
        "ee.pitch": -0.2,
        "ee.yaw": 0.3,
    }

    class FakeTeleop:
        def __init__(self, _config):
            self._actions = iter(
                [
                    {
                        "grip_pos": np.array([0.0, 0.0, 0.0]),
                        "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                        "squeeze": 1.0,
                        "trigger": 0.0,
                    },
                    {
                        "grip_pos": np.array([0.0001, 0.0, 0.0]),
                        "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                        "squeeze": 1.0,
                        "trigger": 0.0,
                    },
                ]
            )
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def get_action(self):
            return next(self._actions)

        def disconnect(self):
            self.is_connected = False

    class FakeRobot:
        name = "jaka_robot"
        is_connected = True

        def get_eef_pose(self):
            return tuple(
                measured_pose[key]
                for key in ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            )

        def servo_enable(self, _enabled, *, representation="joints"):
            pass

    monkeypatch.setattr(xr, "XRController", FakeTeleop)
    monkeypatch.setattr(xr, "_wait_for_xr_controller", lambda _teleop: None)
    bundle = xr.make_xr_device(FakeRobot(), xr.XRControllerConfig())
    bundle["startup"]()

    first = bundle["compute"](measured_pose)
    second = bundle["compute"](measured_pose)

    assert first is not None and second is not None
    assert [second[f"ee.{axis}"] for axis in ("x", "y", "z")] == pytest.approx(
        [first[f"ee.{axis}"] for axis in ("x", "y", "z")]
    )


def test_lock_pose_requires_a_boolean():
    xr = _xr_module()

    with pytest.raises(ValueError, match="lock_pose"):
        xr.XRControllerConfig(lock_pose=[0.0, 0.0, 0.0])


def test_xr_controller_discards_invalid_grip_pose():
    """An enumerated controller with an invalid grip pose must not command the robot."""
    xr = _xr_module()

    class InvalidController:
        is_none = False

        def __getitem__(self, index):
            values = {
                xr.ControllerInputIndex.GRIP_POSITION: np.array([0.4, -0.3, 0.2]),
                xr.ControllerInputIndex.GRIP_ORIENTATION: np.array([0.1, 0.2, 0.3, 0.4]),
                xr.ControllerInputIndex.GRIP_IS_VALID: False,
                xr.ControllerInputIndex.SQUEEZE_VALUE: 1.0,
                xr.ControllerInputIndex.TRIGGER_VALUE: 1.0,
            }
            return values[index]

    controller = object.__new__(xr.XRController)
    controller._external_inputs = None
    controller._is_tracking = False
    controller._step = lambda **_kwargs: {"controller": InvalidController()}

    action = controller.get_action()

    assert controller.is_tracking is False
    assert action["grip_pos"] == pytest.approx([0.0, 0.0, 0.0])
    assert action["grip_quat"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert action["squeeze"] == 0.0
    assert action["trigger"] == 0.0


def test_xr_controller_reports_raw_and_transformed_grip_pose():
    xr = _xr_module()

    class Controller:
        is_none = False

        def __init__(self, position):
            self.position = np.asarray(position, dtype=float)

        def __getitem__(self, index):
            values = {
                xr.ControllerInputIndex.GRIP_POSITION: self.position,
                xr.ControllerInputIndex.GRIP_ORIENTATION: np.array([0.0, 0.0, 0.0, 1.0]),
                xr.ControllerInputIndex.GRIP_IS_VALID: True,
                xr.ControllerInputIndex.SQUEEZE_VALUE: 0.75,
                xr.ControllerInputIndex.TRIGGER_VALUE: 0.25,
            }
            return values[index]

    controller = object.__new__(xr.XRController)
    controller._external_inputs = None
    controller._is_tracking = False
    controller._step = lambda **_kwargs: {
        "controller": Controller([4.0, 5.0, 6.0]),
        "controller_raw": Controller([1.0, 2.0, 3.0]),
    }

    action = controller.get_action()

    assert action["grip_pos"] == pytest.approx([4.0, 5.0, 6.0])
    assert action["raw_grip_pos"] == pytest.approx([1.0, 2.0, 3.0])
    assert action["grip_quat"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert action["raw_grip_quat"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_rpy_conversion_preserves_a_near_gimbal_lock_reference():
    """Equivalent Euler commands must not jump to a distant roll/yaw branch."""
    xr = _xr_module()
    reference = np.array([0.7, math.pi / 2.0, -0.4])
    matrix = xr._pose6_to_base_t_ee((0.0, 0.0, 0.0, *reference))[:3, :3]

    actual = np.asarray(xr._matrix_to_rpy(matrix, reference_rpy=reference))

    assert _wrap_angles(actual - reference) == pytest.approx([0.0, 0.0, 0.0], abs=1e-7)


def test_rpy_conversion_selects_the_nearest_nonprincipal_branch():
    """Crossing pitch 90 degrees must keep the same physical Euler branch."""
    xr = _xr_module()
    reference = np.array([0.7, 2.0, -0.4])
    matrix = xr._pose6_to_base_t_ee((0.0, 0.0, 0.0, *reference))[:3, :3]

    actual = np.asarray(xr._matrix_to_rpy(matrix, reference_rpy=reference))

    assert _wrap_angles(actual - reference) == pytest.approx([0.0, 0.0, 0.0], abs=1e-7)


def test_servo_euler_conversion_preserves_a_near_gimbal_lock_reference():
    """The 8 ms Servo sender must retain the same continuous Euler branch."""
    reference = np.array([0.7, math.pi / 2.0, -0.4])
    quaternion = jaka_driver._euler_to_quaternion(reference)

    actual = jaka_driver._quaternion_to_euler(quaternion, reference_rpy=reference)

    assert _wrap_angles(actual - reference) == pytest.approx([0.0, 0.0, 0.0], abs=1e-7)
