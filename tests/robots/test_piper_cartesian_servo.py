from __future__ import annotations

import math

import numpy as np
import pytest

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


def make_servo(**overrides):
    config = PiperCartesianServoConfig(**overrides)
    return PiperCartesianServo(config, joint_limits=LIMITS, fk_tcp=wrist_fk)


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
