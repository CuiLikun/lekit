import importlib
import math

import numpy as np
import pytest

from lekit.robots.jaka_robot import jaka_robot as jaka_driver


def _xr_module():
    return importlib.import_module("examples.isaac_teleop_to_jaka.xr")


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


def test_default_xr_base_transform_matches_jaka_operator_axes():
    """Right/forward/up controller motion must map to the matching robot axes.

    Conventions:
      - OpenXR controller: X=Right, Y=Up, Z=Backward (toward operator).
      - JAKA base frame:   X=Right, Y=Forward, Z=Up.
    """
    xr = _xr_module()
    rotation = np.asarray(xr.XRControllerConfig().base_T_anchor, dtype=float)[:3, :3]
    yaw = np.deg2rad(xr.DEFAULT_OPERATOR_YAW_DEG)
    operator_right = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    operator_forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])

    # Right -> robot +X, forward -> robot +Y, up -> robot +Z.
    assert rotation @ operator_right == pytest.approx([1.0, 0.0, 0.0])
    assert rotation @ operator_forward == pytest.approx([0.0, 1.0, 0.0])
    assert rotation @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])
    assert rotation.T @ rotation == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_operator_yaw_generates_requested_transform():
    """Rotating the operator CCW by ``yaw`` in OpenXR's horizontal plane keeps the
    operator's right direction aligned with the robot's right axis (+X)."""
    xr = _xr_module()
    yaw = np.deg2rad(33.635)
    rotation = np.asarray(xr.XRControllerConfig(operator_yaw_deg=33.635).base_T_anchor)[:3, :3]
    # The OpenXR +X axis rotated CCW by `yaw` around OpenXR +Y: still the operator's
    # right direction expressed in OpenXR's frame; must map to robot right (+X).
    assert rotation @ np.array([np.cos(yaw), 0.0, np.sin(yaw)]) == pytest.approx([1.0, 0.0, 0.0])


def test_default_cartesian_nlf_uses_a5_balanced_profile():
    xr = _xr_module()

    config = xr.XRControllerConfig()
    assert config.servo_linear_velocity_m_s == 0.2
    assert config.servo_linear_jerk_m_s3 == 3.0
    assert config.servo_angular_jerk_rad_s3 == 8.0


def test_static_calibration_decouples_captured_horizontal_motion():
    """Replay the right/forward vectors captured from the physical XR workstation."""
    xr = _xr_module()
    config = xr.XRControllerConfig(use_head_yaw=False, operator_yaw_deg=-10.0364)
    rotation = np.asarray(config.base_T_anchor, dtype=float)[:3, :3]
    raw_right = np.array([0.14900, -0.00965, -0.02315])
    raw_forward = np.array([-0.03119, 0.03398, -0.15691])

    mapped_right = rotation @ raw_right
    mapped_forward = rotation @ raw_forward

    assert mapped_right[0] > 0.0
    assert abs(mapped_right[1] / mapped_right[0]) < 0.03
    assert mapped_forward[1] > 0.0
    assert abs(mapped_forward[0] / mapped_forward[1]) < 0.03


def test_calibrated_station_motion_maps_to_matching_tcp_axes(monkeypatch):
    """Exercise the raw OpenXR -> transformed grip -> clutch -> TCP target path."""
    xr = _xr_module()
    station_yaw = np.deg2rad(-10.0364)
    # Keep the diagonal transition between the two probes below MAX_EE_STEP_M.
    delta = 0.03
    raw_poses = iter(
        [
            np.zeros(3),
            delta * np.array([np.sin(station_yaw), 0.0, -np.cos(station_yaw)]),
            delta * np.array([np.cos(station_yaw), 0.0, np.sin(station_yaw)]),
        ]
    )

    class FakeTeleop:
        def __init__(self, config):
            self._rotation = np.asarray(config.base_T_anchor, dtype=float)[:3, :3]
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def get_action(self):
            raw_pos = next(raw_poses)
            return {
                "grip_pos": self._rotation @ raw_pos,
                "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "raw_grip_pos": raw_pos,
                "raw_grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "squeeze": 1.0,
                "trigger": 0.0,
            }

        def disconnect(self):
            self.is_connected = False

    home = {
        "ee.x": 0.4,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.roll": 0.1,
        "ee.pitch": -0.2,
        "ee.yaw": 0.3,
    }

    class FakeRobot:
        name = "jaka_robot"
        is_connected = True

        def get_eef_pose(self):
            keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            return tuple(home[key] for key in keys)

        def servo_enable(self, _enabled, *, representation="joints"):
            pass

    monkeypatch.setattr(xr, "XRController", FakeTeleop)
    monkeypatch.setattr(xr, "_wait_for_xr_controller", lambda _teleop: None)
    config = xr.XRControllerConfig(
        lock_pose=True, use_head_yaw=False, operator_yaw_deg=-10.0364
    )
    bundle = xr.make_xr_device(FakeRobot(), config)
    bundle["startup"]()

    origin = bundle["compute"](home)
    forward = bundle["compute"](home)
    right = bundle["compute"](home)

    assert origin is not None and forward is not None and right is not None
    origin_pos = np.array([origin[f"ee.{axis}"] for axis in ("x", "y", "z")])
    forward_pos = np.array([forward[f"ee.{axis}"] for axis in ("x", "y", "z")])
    right_pos = np.array([right[f"ee.{axis}"] for axis in ("x", "y", "z")])
    assert forward_pos - origin_pos == pytest.approx([0.0, delta, 0.0], abs=1e-7)
    assert right_pos - origin_pos == pytest.approx([delta, 0.0, 0.0], abs=1e-7)


def test_head_relative_mapping_tracks_operator_heading_at_each_engage(monkeypatch):
    """Body-forward motion must remain TCP +Y after the operator turns 90 degrees."""
    xr = _xr_module()
    delta = 0.03
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    head_turned_left = np.array([0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])
    samples = iter(
        [
            (np.zeros(3), identity, 1.0),
            (np.array([0.0, 0.0, -delta]), identity, 1.0),
            (np.zeros(3), identity, 0.0),
            (np.zeros(3), head_turned_left, 1.0),
            (np.array([delta, 0.0, 0.0]), head_turned_left, 1.0),
        ]
    )

    class FakeTeleop:
        def __init__(self, config):
            self._rotation = np.asarray(config.base_T_anchor, dtype=float)[:3, :3]
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def get_action(self):
            raw_pos, head_quat, squeeze = next(samples)
            return {
                "grip_pos": self._rotation @ raw_pos,
                "grip_quat": identity,
                "raw_grip_pos": raw_pos,
                "raw_grip_quat": identity,
                "head_quat": head_quat,
                "head_is_tracking": True,
                "squeeze": squeeze,
                "trigger": 0.0,
            }

        def disconnect(self):
            self.is_connected = False

    home = {
        "ee.x": 0.4,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.roll": 0.1,
        "ee.pitch": -0.2,
        "ee.yaw": 0.3,
    }

    class FakeRobot:
        name = "jaka_robot"
        is_connected = True

        def get_eef_pose(self):
            keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            return tuple(home[key] for key in keys)

        def servo_enable(self, _enabled, *, representation="joints"):
            pass

    monkeypatch.setattr(xr, "XRController", FakeTeleop)
    monkeypatch.setattr(xr, "_wait_for_xr_controller", lambda _teleop: None)
    bundle = xr.make_xr_device(FakeRobot(), xr.XRControllerConfig(lock_pose=True))
    bundle["startup"]()

    origin_facing_anchor = bundle["compute"](home)
    forward_facing_anchor = bundle["compute"](home)
    assert bundle["compute"](home) is None
    assert bundle["telemetry"]["clutch_released"] is True
    origin_after_turn = bundle["compute"](home)
    assert bundle["telemetry"]["clutch_released"] is False
    forward_after_turn = bundle["compute"](home)

    assert all(
        action is not None
        for action in (
            origin_facing_anchor,
            forward_facing_anchor,
            origin_after_turn,
            forward_after_turn,
        )
    )
    for origin, forward in (
        (origin_facing_anchor, forward_facing_anchor),
        (origin_after_turn, forward_after_turn),
    ):
        origin_pos = np.array([origin[f"ee.{axis}"] for axis in ("x", "y", "z")])
        forward_pos = np.array([forward[f"ee.{axis}"] for axis in ("x", "y", "z")])
        assert forward_pos - origin_pos == pytest.approx([0.0, delta, 0.0], abs=1e-7)


def test_head_relative_mapping_waits_for_valid_head_tracking(monkeypatch):
    """A held squeeze may engage only after a valid headset heading is available."""
    xr = _xr_module()
    delta = 0.03
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    samples = iter(
        [
            (np.zeros(3), False),
            (np.zeros(3), True),
            (np.array([0.0, 0.0, -delta]), True),
        ]
    )

    class FakeTeleop:
        def __init__(self, _config):
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def get_action(self):
            raw_pos, head_is_tracking = next(samples)
            return {
                "grip_pos": raw_pos,
                "grip_quat": identity,
                "raw_grip_pos": raw_pos,
                "raw_grip_quat": identity,
                "head_quat": identity,
                "head_is_tracking": head_is_tracking,
                "squeeze": 1.0,
                "trigger": 0.0,
            }

        def disconnect(self):
            self.is_connected = False

    home = {
        "ee.x": 0.4,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.roll": 0.1,
        "ee.pitch": -0.2,
        "ee.yaw": 0.3,
    }

    class FakeRobot:
        name = "jaka_robot"
        is_connected = True

        def get_eef_pose(self):
            keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            return tuple(home[key] for key in keys)

        def servo_enable(self, _enabled, *, representation="joints"):
            pass

    monkeypatch.setattr(xr, "XRController", FakeTeleop)
    monkeypatch.setattr(xr, "_wait_for_xr_controller", lambda _teleop: None)
    bundle = xr.make_xr_device(FakeRobot(), xr.XRControllerConfig(lock_pose=True))
    bundle["startup"]()

    assert bundle["compute"](home) is None
    assert bundle["telemetry"]["clutch_engaged"] is False

    origin = bundle["compute"](home)
    forward = bundle["compute"](home)

    assert origin is not None and forward is not None
    assert bundle["telemetry"]["clutch_engaged"] is True
    origin_pos = np.array([origin[f"ee.{axis}"] for axis in ("x", "y", "z")])
    forward_pos = np.array([forward[f"ee.{axis}"] for axis in ("x", "y", "z")])
    assert forward_pos - origin_pos == pytest.approx([0.0, delta, 0.0], abs=1e-7)


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
    config = xr.XRControllerConfig(lock_pose=True, use_head_yaw=False)
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


def test_thumbstick_trims_locked_tcp_orientation_without_center_drift(monkeypatch):
    xr = _xr_module()
    measured_pose = {
        "ee.x": 0.4,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.roll": 0.1,
        "ee.pitch": -0.2,
        "ee.yaw": 0.3,
    }
    actions = iter(
        [
            {"thumbstick_x": 0.0, "thumbstick_y": 0.0, "thumbstick_click": 0.0},
            {"thumbstick_x": 1.0, "thumbstick_y": 1.0, "thumbstick_click": 0.0},
            {"thumbstick_x": 1.0, "thumbstick_y": 0.0, "thumbstick_click": 1.0},
            {"thumbstick_x": 0.1, "thumbstick_y": -0.1, "thumbstick_click": 0.0},
        ]
    )

    class FakeTeleop:
        def __init__(self, _config):
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def get_action(self):
            return {
                "grip_pos": np.zeros(3),
                "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "squeeze": 1.0,
                "trigger": 0.0,
                **next(actions),
            }

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

    times = iter([10.0, 10.2, 10.3, 10.4])
    monkeypatch.setattr(xr, "XRController", FakeTeleop)
    monkeypatch.setattr(xr, "_wait_for_xr_controller", lambda _teleop: None)
    monkeypatch.setattr(xr.time, "monotonic", lambda: next(times))
    config = xr.XRControllerConfig(
        lock_pose=True,
        use_head_yaw=False,
        thumbstick_deadband=0.15,
        thumbstick_angular_speed_rad_s=0.5,
    )
    bundle = xr.make_xr_device(FakeRobot(), config)
    bundle["startup"]()

    origin = bundle["compute"](measured_pose)
    pitch_yaw = bundle["compute"](measured_pose)
    roll = bundle["compute"](measured_pose)
    centered = bundle["compute"](measured_pose)

    assert origin is not None and pitch_yaw is not None and roll is not None and centered is not None
    assert [origin[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw")] == pytest.approx(
        [0.1, -0.2, 0.3]
    )
    # Elapsed time is capped at 0.1 s, so full deflection adds 0.05 rad per frame.
    assert [pitch_yaw[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw")] == pytest.approx(
        [0.1, -0.15, 0.35]
    )
    assert [roll[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw")] == pytest.approx(
        [0.15, -0.15, 0.35]
    )
    assert [centered[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw")] == pytest.approx(
        [0.15, -0.15, 0.35]
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
    bundle = xr.make_xr_device(FakeRobot(), xr.XRControllerConfig(use_head_yaw=False))
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"thumbstick_deadband": 1.0}, "thumbstick_deadband"),
        ({"thumbstick_deadband": -0.1}, "thumbstick_deadband"),
        ({"thumbstick_angular_speed_rad_s": 0.0}, "thumbstick_angular_speed_rad_s"),
    ],
)
def test_thumbstick_config_rejects_unsafe_values(kwargs, message):
    xr = _xr_module()

    with pytest.raises(ValueError, match=message):
        xr.XRControllerConfig(**kwargs)


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
    assert action["a_button"] == 0.0
    assert action["b_button"] == 0.0
    assert action["thumbstick_x"] == 0.0
    assert action["thumbstick_y"] == 0.0
    assert action["thumbstick_click"] == 0.0


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
                xr.ControllerInputIndex.PRIMARY_CLICK: 1.0,
                xr.ControllerInputIndex.SECONDARY_CLICK: 0.5,
                xr.ControllerInputIndex.THUMBSTICK_X: -0.75,
                xr.ControllerInputIndex.THUMBSTICK_Y: 0.6,
                xr.ControllerInputIndex.THUMBSTICK_CLICK: 1.0,
            }
            return values[index]

    class Head:
        is_none = False

        def __getitem__(self, index):
            values = {
                xr.HeadPoseIndex.ORIENTATION: np.array([0.0, 0.5, 0.0, math.sqrt(0.75)]),
                xr.HeadPoseIndex.IS_VALID: True,
            }
            return values[index]

    controller = object.__new__(xr.XRController)
    controller._external_inputs = None
    controller._is_tracking = False
    controller._step = lambda **_kwargs: {
        "controller": Controller([4.0, 5.0, 6.0]),
        "controller_raw": Controller([1.0, 2.0, 3.0]),
        "head": Head(),
    }

    action = controller.get_action()

    assert action["grip_pos"] == pytest.approx([4.0, 5.0, 6.0])
    assert action["raw_grip_pos"] == pytest.approx([1.0, 2.0, 3.0])
    assert action["grip_quat"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert action["raw_grip_quat"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert action["head_quat"] == pytest.approx([0.0, 0.5, 0.0, math.sqrt(0.75)])
    assert action["head_is_tracking"] is True
    assert action["a_button"] == 1.0
    assert action["b_button"] == 0.5
    assert action["thumbstick_x"] == pytest.approx(-0.75)
    assert action["thumbstick_y"] == pytest.approx(0.6)
    assert action["thumbstick_click"] == 1.0


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
