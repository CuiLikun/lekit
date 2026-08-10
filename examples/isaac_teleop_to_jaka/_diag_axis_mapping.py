"""Standalone diagnostic: move the controller +5 cm along each operator axis and print
the resulting JAKA ``ee.{x,y,z}`` target delta. Run with:

    python -m examples.isaac_teleop_to_jaka._diag_axis_mapping

It uses the same ``make_xr_device`` plumbing and the same ``base_T_anchor`` matrix the
recorder sends through ``servo_p`` — no hardware, no Isaac Teleop session required.
The fake XR controller applies ``R @ p + t`` internally so the matrix is exercised
exactly as the real SDK's ``ControllerTransform`` does.

The expected JAKA convention is:
    X = Right, Y = Forward, Z = Up.

The expected OpenXR grip-pose convention (per OpenXR spec, CloudXR, and the
SO101 example's docstring) is:
    X = Right, Y = Up, Z = Backward (toward operator).

For the configured operator yaw, the diagnostic expresses physical right/forward in
the OpenXR anchor frame and verifies the intuitive mapping:
    hand right   -> JAKA +X (robot right)
    hand forward -> JAKA +Y (robot forward)
    hand up      -> JAKA +Z (robot up)
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

from examples.isaac_teleop_to_jaka.xr import XRControllerConfig, make_xr_device  # noqa: E402

# Measured EE pose — what the JAKA arm reports back. Any finite pose works for the
# diagnostic; only the *delta* between successive frames is meaningful.
HOME_POSE = (0.40, -0.10, 0.30, 0.10, -0.20, 0.30)
HOME_DICT = {
    "ee.x": HOME_POSE[0],
    "ee.y": HOME_POSE[1],
    "ee.z": HOME_POSE[2],
    "ee.roll": HOME_POSE[3],
    "ee.pitch": HOME_POSE[4],
    "ee.yaw": HOME_POSE[5],
}


class FakeRobot:
    name = "jaka_robot"
    is_connected = True

    def get_eef_pose(self):
        return HOME_POSE

    def get_observation(self):
        return dict(HOME_DICT)

    def servo_enable(self, _enabled, *, representation="eef"):
        pass

    def is_in_servo(self):
        return True


class FakeXRController:
    """Reports a configurable OpenXR pose each time ``get_action()`` is called.

    Applies the same ``base_T_anchor`` transform the real Isaac Teleop SDK applies
    via ``controllers.transformed(...)`` (``R @ p + t``), so the diagnostic exercises
    the same matrix the recorder sends through ``servo_p``.
    """

    _instance: FakeXRController | None = None

    def __init__(self, _config):
        self._raw_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._base_T_anchor = np.asarray(_config.base_T_anchor, dtype=np.float32)
        self.is_connected = True
        FakeXRController._instance = self

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def set_raw_pose(self, raw_pose: np.ndarray) -> None:
        self._raw_pose = np.asarray(raw_pose, dtype=np.float32)

    def get_action(self):
        # Mirror the SDK's transform_utils.transform_position: rotation @ p + translation.
        rotation = self._base_T_anchor[:3, :3]
        translation = self._base_T_anchor[:3, 3]
        transformed = (
            rotation @ self._raw_pose.astype(np.float64) + translation
        ).astype(np.float32)
        return {
            "grip_pos": transformed,
            "grip_quat": self._quat.copy(),
            "raw_grip_pos": self._raw_pose.copy(),
            "raw_grip_quat": self._quat.copy(),
            "head_quat": self._quat.copy(),
            "head_is_tracking": True,
            "squeeze": 1.0,  # keep clutch engaged the whole time
            "trigger": 0.0,
        }


def main() -> None:
    with patch("examples.isaac_teleop_to_jaka.xr.XRController", FakeXRController), \
         patch("examples.isaac_teleop_to_jaka.xr._wait_for_xr_controller", lambda _t: None):
        config = XRControllerConfig(lock_pose=True)  # lock_pose keeps orientation fixed
        bundle = make_xr_device(FakeRobot(), config)
        bundle["startup"]()
        compute = bundle["compute"]

        delta = 0.05  # 5 cm push per axis

        # Express the operator's physical horizontal axes in the OpenXR anchor frame.
        yaw = np.deg2rad(config.operator_yaw_deg)
        operator_right = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        operator_forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])

        # For an operator behind the robot, intuitive mapping:
        #   hand right  -> robot right  (+X)  i.e. ee.x should INCREASE
        #   hand forward-> robot forward (+Y)  i.e. ee.y should INCREASE
        #   hand up     -> robot up     (+Z)  i.e. ee.z should INCREASE
        cases = [
            ("physical hand right", delta * operator_right, ("ee.x", "increase")),
            ("physical hand forward", delta * operator_forward, ("ee.y", "increase")),
            ("physical hand up", np.array([0.0, +delta, 0.0]), ("ee.z", "increase")),
        ]

        fake = FakeXRController._instance
        assert fake is not None

        print(f"base_T_anchor rotation = {np.asarray(config.base_T_anchor)[:3, :3].tolist()}")
        print()
        print(f"{'input':<28} {'Δee.x':>8} {'Δee.y':>8} {'Δee.z':>8}   expected")
        print("-" * 72)

        for label, raw_delta, expected in cases:
            # Reset the controller and re-engage the clutch for each test so the
            # per-frame MAX_EE_STEP_M clamp doesn't accumulate deadband across
            # iterations.
            fake.set_raw_pose(np.zeros(3, dtype=np.float32))
            origin_target = compute(HOME_DICT)
            assert origin_target is not None, "clutch failed to re-engage"
            fake.set_raw_pose(raw_delta)
            target = compute(HOME_DICT)
            dx = target["ee.x"] - origin_target["ee.x"]
            dy = target["ee.y"] - origin_target["ee.y"]
            dz = target["ee.z"] - origin_target["ee.z"]
            print(f"{label:<28} {dx:+8.4f} {dy:+8.4f} {dz:+8.4f}   {expected[1]} {expected[0]}")


if __name__ == "__main__":
    main()
