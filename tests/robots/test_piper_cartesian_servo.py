from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lekit.robots.piper.cartesian_servo import (
    PiperCartesianServo,
    PiperCartesianServoConfig,
    _norm_bounded,
)

LIMITS = tuple((-math.pi, math.pi) for _ in range(6))


def wrist_fk(joints):
    q = np.asarray(joints, dtype=float)
    return np.array(
        [
            q[0],
            q[1],
            q[2],
            q[3] + q[5],
            math.sin(q[4]) * (q[3] - q[5]),
            q[4],
        ],
        dtype=float,
    )


def make_servo(*, joint_limits=LIMITS, fk_tcp=wrist_fk, **overrides):
    config = PiperCartesianServoConfig(**overrides)
    return PiperCartesianServo(config, joint_limits=joint_limits, fk_tcp=fk_tcp)


def test_well_conditioned_step_reduces_full_pose_error():
    servo = make_servo(max_joint_acceleration_rad_s2=100.0, max_joint_jerk_rad_s3=1000.0)
    joints = np.array([0.1, -0.1, 0.2, 0.2, 0.5, -0.1])
    measured = wrist_fk(joints)
    target = measured + np.array([0.001, -0.001, 0.001, 0.01, -0.01, 0.01])

    step = servo.step(joints, measured, target, dt=1.0 / 30.0)

    assert np.linalg.norm(wrist_fk(step.joint_target) - target) < np.linalg.norm(measured - target)
    assert step.diagnostics.orientation_scale > 0.9


def test_exact_wrist_singularity_returns_bounded_continuous_target():
    servo = make_servo(max_joint_acceleration_rad_s2=100.0, max_joint_jerk_rad_s3=1000.0)
    joints = np.array([0.0, 0.0, 0.0, 0.2, 0.0, -0.2])
    measured = wrist_fk(joints)
    target = measured + np.array([0.002, 0.0, 0.0, 0.0, 0.10, 0.0])

    step = servo.step(joints, measured, target, dt=1.0 / 30.0)

    velocity = (np.asarray(step.joint_target) - joints) * 30.0
    assert np.max(np.abs(velocity)) <= servo.config.max_joint_velocity_rad_s + 1e-9
    assert step.joint_target[0] > joints[0]
    assert step.diagnostics.orientation_scale == pytest.approx(0.15, abs=0.02)
    assert step.diagnostics.position_error_m == pytest.approx(0.002)


def test_three_axis_linear_task_velocity_is_norm_limited():
    requested = np.array([1.0, 1.0, 1.0])

    bounded = _norm_bounded(requested, 0.10)

    assert np.linalg.norm(bounded) == pytest.approx(0.10)
    assert np.all(bounded < requested)


def test_three_axis_angular_task_velocity_is_norm_limited():
    requested = np.array([1.0, 1.0, 1.0])

    bounded = _norm_bounded(requested, 0.50)

    assert np.linalg.norm(bounded) == pytest.approx(0.50)
    assert np.all(bounded < requested)


def test_repeated_steps_obey_velocity_acceleration_and_jerk_limits():
    servo = make_servo(
        max_joint_velocity_rad_s=0.4,
        max_joint_acceleration_rad_s2=0.8,
        max_joint_jerk_rad_s3=4.0,
    )
    dt = 1.0 / 30.0
    joints = np.array([0.0, 0.0, 0.0, 0.2, 0.35, -0.2])
    target = wrist_fk(joints) + np.array([0.05, -0.04, 0.03, 0.4, -0.3, 0.2])
    velocities = []
    accelerations = []

    for _ in range(20):
        step = servo.step(joints, wrist_fk(joints), target, dt=dt)
        next_joints = np.asarray(step.joint_target)
        velocity = (next_joints - joints) / dt
        velocities.append(velocity)
        if len(velocities) > 1:
            accelerations.append((velocities[-1] - velocities[-2]) / dt)
        joints = next_joints

    assert np.max(np.abs(velocities)) <= 0.4 + 1e-8
    assert np.max(np.abs(accelerations)) <= 0.8 + 1e-8
    jerks = np.diff(np.asarray(accelerations), axis=0) / dt
    assert np.max(np.abs(jerks)) <= 4.0 + 1e-8


def test_reset_seeds_zero_motion_without_first_frame_jump():
    servo = make_servo()
    joints = np.array([0.0, 0.0, 0.0, 0.1, 0.4, -0.1])
    pose = wrist_fk(joints)
    servo.step(joints, pose, pose + np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), dt=1 / 30)

    servo.reset()
    held = servo.step(joints, pose, pose, dt=1 / 30)

    assert held.joint_target == pytest.approx(joints)


def test_so3_wraparound_uses_the_short_orientation_error():
    servo = make_servo(max_joint_acceleration_rad_s2=100.0, max_joint_jerk_rad_s3=1000.0)
    joints = np.array([0.0, 0.0, 0.0, -1.56, 0.5, -1.571592653589793])
    measured = wrist_fk(joints)
    target = measured.copy()
    target[3] = math.pi - 0.01

    step = servo.step(joints, measured, target, dt=1 / 30)
    error = (
        Rotation.from_euler("xyz", target[3:])
        * Rotation.from_euler("xyz", wrist_fk(step.joint_target)[3:]).inv()
    ).magnitude()

    assert step.diagnostics.orientation_error_rad == pytest.approx(0.02, abs=1e-6)
    assert error < 0.02


def test_repeated_singular_steps_keep_j4_and_j6_motion_continuous():
    servo = make_servo(max_joint_acceleration_rad_s2=100.0, max_joint_jerk_rad_s3=1000.0)
    joints = np.array([0.0, 0.0, 0.0, 0.2, 0.0, -0.2])
    target = wrist_fk(joints) + np.array([0.002, 0.0, 0.0, 0.0, 0.10, 0.0])
    wrist_velocities = []

    for _ in range(8):
        step = servo.step(joints, wrist_fk(joints), target, dt=1 / 30)
        next_joints = np.asarray(step.joint_target)
        wrist_velocities.append((next_joints[[3, 5]] - joints[[3, 5]]) * 30.0)
        joints = next_joints

    wrist_velocities = np.asarray(wrist_velocities)
    assert np.all(wrist_velocities[1:] * wrist_velocities[:-1] >= -1e-9)
    assert np.max(np.abs(wrist_velocities)) <= servo.config.max_joint_velocity_rad_s + 1e-8


def test_orientation_target_catches_up_after_j5_leaves_singularity():
    servo = make_servo(max_joint_acceleration_rad_s2=100.0, max_joint_jerk_rad_s3=1000.0)
    joints = np.array([0.0, 0.0, 0.0, 0.2, 0.0, -0.2])
    target = wrist_fk(joints) + np.array([0.0, 0.0, 0.0, 0.0, 0.10, 0.0])

    singular = servo.step(joints, wrist_fk(joints), target, dt=1 / 30)
    joints[4] = 0.5
    before = np.linalg.norm(wrist_fk(joints)[3:] - target[3:])
    for _ in range(10):
        step = servo.step(joints, wrist_fk(joints), target, dt=1 / 30)
        joints = np.asarray(step.joint_target)

    assert singular.diagnostics.orientation_scale < 0.2
    assert step.diagnostics.orientation_scale > 0.9
    assert np.linalg.norm(wrist_fk(joints)[3:] - target[3:]) < before


def test_soft_joint_limit_attenuates_outward_motion_but_not_inward_motion():
    limits = ((-1.0, 1.0),) + LIMITS[1:]
    outward_joints = np.array([0.9, 0.0, 0.0, 0.2, 0.5, -0.2])
    inward_joints = outward_joints.copy()
    outward = make_servo(
        joint_limits=limits,
        joint_limit_margin_rad=0.2,
        max_joint_acceleration_rad_s2=100.0,
        max_joint_jerk_rad_s3=1000.0,
    )
    inward = make_servo(
        joint_limits=limits,
        joint_limit_margin_rad=0.2,
        max_joint_acceleration_rad_s2=100.0,
        max_joint_jerk_rad_s3=1000.0,
    )

    outward_step = outward.step(
        outward_joints, wrist_fk(outward_joints), wrist_fk(outward_joints) + np.array([1.0, 0, 0, 0, 0, 0]), dt=1 / 30
    )
    inward_step = inward.step(
        inward_joints, wrist_fk(inward_joints), wrist_fk(inward_joints) - np.array([1.0, 0, 0, 0, 0, 0]), dt=1 / 30
    )
    outward_velocity = (outward_step.joint_target[0] - outward_joints[0]) * 30.0
    inward_velocity = (inward_step.joint_target[0] - inward_joints[0]) * 30.0

    assert outward_step.diagnostics.joint_limit_saturated
    assert outward_velocity <= 0.5 * outward.config.max_joint_velocity_rad_s + 1e-8
    assert abs(inward_velocity) > abs(outward_velocity)


def test_near_joint_limit_sets_saturation_without_exceeding_hard_limit():
    limits = ((-1.0, 1.0),) + LIMITS[1:]
    joints = np.array([0.999, 0.0, 0.0, 0.2, 0.5, -0.2])
    servo = make_servo(
        joint_limits=limits,
        joint_limit_margin_rad=0.2,
        max_joint_acceleration_rad_s2=100.0,
        max_joint_jerk_rad_s3=1000.0,
    )

    step = servo.step(joints, wrist_fk(joints), wrist_fk(joints) + np.array([1.0, 0, 0, 0, 0, 0]), dt=1 / 30)

    assert step.diagnostics.joint_limit_saturated
    assert -1.0 <= step.joint_target[0] <= 1.0


@pytest.mark.parametrize(
    ("measured_joints", "measured_tcp", "target_tcp", "dt"),
    [
        ([0.0] * 5, [0.0] * 6, [0.0] * 6, 1 / 30),
        ([0.0] * 6, [0.0] * 5, [0.0] * 6, 1 / 30),
        ([0.0] * 6, [0.0] * 6, [math.nan] + [0.0] * 5, 1 / 30),
        ([0.0] * 6, [0.0] * 6, [0.0] * 6, 0.0),
        ([0.0] * 6, [0.0] * 6, [0.0] * 6, -1 / 30),
    ],
)
def test_step_rejects_invalid_inputs(measured_joints, measured_tcp, target_tcp, dt):
    servo = make_servo()

    with pytest.raises(ValueError):
        servo.step(measured_joints, measured_tcp, target_tcp, dt=dt)


def test_step_rejects_malformed_fk_pose():
    servo = make_servo(fk_tcp=lambda _joints: [0.0] * 5)

    with pytest.raises(ValueError, match="FK TCP pose"):
        servo.step([0.0] * 6, [0.0] * 6, [0.0] * 6, dt=1 / 30)
