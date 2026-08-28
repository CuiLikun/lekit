from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PiperCartesianServoConfig:
    position_gain_s: float = 4.0
    orientation_gain_s: float = 4.0
    max_tcp_velocity_m_s: float = 0.10
    max_tcp_angular_velocity_rad_s: float = 0.50
    max_joint_velocity_rad_s: float = 0.50
    max_joint_acceleration_rad_s2: float = 1.50
    max_joint_jerk_rad_s3: float = 8.0
    joint_limit_margin_rad: float = 0.17453292519943295
    jacobian_step_rad: float = 1e-4
    characteristic_length_m: float = 0.25
    singular_value_low: float = 0.03
    singular_value_high: float = 0.12
    minimum_orientation_scale: float = 0.15
    minimum_damping: float = 0.005
    maximum_damping: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.position_gain_s,
            self.orientation_gain_s,
            self.max_tcp_velocity_m_s,
            self.max_tcp_angular_velocity_rad_s,
            self.max_joint_velocity_rad_s,
            self.max_joint_acceleration_rad_s2,
            self.max_joint_jerk_rad_s3,
            self.joint_limit_margin_rad,
            self.jacobian_step_rad,
            self.characteristic_length_m,
            self.singular_value_low,
            self.singular_value_high,
            self.minimum_orientation_scale,
            self.minimum_damping,
            self.maximum_damping,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all servo configuration values must be finite")
        positive = values[:10]
        if not all(value > 0.0 for value in positive):
            raise ValueError("servo gains and limits must be positive")
        if not 0.0 < self.minimum_orientation_scale <= 1.0:
            raise ValueError("minimum_orientation_scale must be in (0, 1]")
        if not 0.0 < self.minimum_damping <= self.maximum_damping:
            raise ValueError("damping must satisfy 0 < minimum_damping <= maximum_damping")
        if not 0.0 <= self.singular_value_low < self.singular_value_high:
            raise ValueError("singular values must satisfy 0 <= low < high")


@dataclass(frozen=True)
class PiperCartesianServoDiagnostics:
    state: str
    minimum_singular_value: float
    damping: float
    orientation_scale: float
    position_error_m: float
    orientation_error_rad: float
    maximum_joint_velocity_rad_s: float
    solver_duration_ms: float
    joint_limit_saturated: bool


@dataclass(frozen=True)
class PiperCartesianServoStep:
    joint_target: tuple[float, float, float, float, float, float]
    diagnostics: PiperCartesianServoDiagnostics


def _validated_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return vector.copy()


def _damped_pinv(matrix: np.ndarray, damping: float) -> np.ndarray:
    return matrix.T @ np.linalg.solve(
        matrix @ matrix.T + (damping * damping) * np.eye(matrix.shape[0]),
        np.eye(matrix.shape[0]),
    )


class PiperCartesianServo:
    def __init__(
        self,
        config: PiperCartesianServoConfig,
        *,
        joint_limits: Sequence[tuple[float, float]],
        fk_tcp: Callable[[Sequence[float]], Sequence[float]],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, PiperCartesianServoConfig):
            raise TypeError("config must be PiperCartesianServoConfig")
        if len(joint_limits) != 6:
            raise ValueError("joint_limits must contain six pairs")
        validated_limits = []
        for lower, upper in joint_limits:
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("joint limits must be finite pairs with lower < upper")
            validated_limits.append((float(lower), float(upper)))
        self.config = config
        self.joint_limits = tuple(validated_limits)
        self.fk_tcp = fk_tcp
        self.monotonic = monotonic

    def reset(self) -> None:
        pass

    def step(
        self,
        measured_joints: Sequence[float],
        measured_tcp: Sequence[float],
        target_tcp: Sequence[float],
        *,
        dt: float,
    ) -> PiperCartesianServoStep:
        started = self.monotonic()
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        joints = _validated_vector(measured_joints, 6, "measured joints")
        measured = _validated_vector(measured_tcp, 6, "measured TCP pose")
        target = _validated_vector(target_tcp, 6, "target TCP pose")
        position_error = target[:3] - measured[:3]
        orientation_error = (
            Rotation.from_euler("xyz", target[3:]) * Rotation.from_euler("xyz", measured[3:]).inv()
        ).as_rotvec()

        jacobian = np.empty((6, 6), dtype=float)
        for index in range(6):
            plus = joints.copy()
            minus = joints.copy()
            plus[index] += self.config.jacobian_step_rad
            minus[index] -= self.config.jacobian_step_rad
            plus_pose = _validated_vector(self.fk_tcp(plus), 6, "FK TCP pose")
            minus_pose = _validated_vector(self.fk_tcp(minus), 6, "FK TCP pose")
            jacobian[:3, index] = (plus_pose[:3] - minus_pose[:3]) / (2.0 * self.config.jacobian_step_rad)
            jacobian[3:, index] = (
                Rotation.from_euler("xyz", plus_pose[3:]) * Rotation.from_euler("xyz", minus_pose[3:]).inv()
            ).as_rotvec() / (2.0 * self.config.jacobian_step_rad)

        condition_jacobian = jacobian.copy()
        condition_jacobian[:3] /= self.config.characteristic_length_m
        sigma_min = float(np.linalg.svd(condition_jacobian, compute_uv=False)[-1])
        t = np.clip(
            (sigma_min - self.config.singular_value_low)
            / (self.config.singular_value_high - self.config.singular_value_low),
            0.0,
            1.0,
        )
        t = t * t * (3.0 - 2.0 * t)
        orientation_scale = self.config.minimum_orientation_scale + (1.0 - self.config.minimum_orientation_scale) * t
        damping = self.config.maximum_damping + (self.config.minimum_damping - self.config.maximum_damping) * t

        bounded_position_velocity = np.clip(
            self.config.position_gain_s * position_error,
            -self.config.max_tcp_velocity_m_s,
            self.config.max_tcp_velocity_m_s,
        )
        bounded_rotation_velocity = np.clip(
            self.config.orientation_gain_s * orientation_error,
            -self.config.max_tcp_angular_velocity_rad_s,
            self.config.max_tcp_angular_velocity_rad_s,
        )
        j_position = jacobian[:3]
        j_rotation = jacobian[3:]
        pinv_position = _damped_pinv(j_position, damping)
        qdot_position = pinv_position @ bounded_position_velocity
        null_position = np.eye(6) - pinv_position @ j_position
        j_rotation_remaining = j_rotation @ null_position
        qdot_rotation = _damped_pinv(j_rotation_remaining, damping) @ (
            bounded_rotation_velocity * orientation_scale - j_rotation @ qdot_position
        )
        qdot = qdot_position + null_position @ qdot_rotation
        qdot = np.clip(qdot, -self.config.max_joint_velocity_rad_s, self.config.max_joint_velocity_rad_s)
        unclipped_target = joints + qdot * dt
        joint_target_array = np.array(
            [
                np.clip(value, lower, upper)
                for value, (lower, upper) in zip(unclipped_target, self.joint_limits, strict=True)
            ],
            dtype=float,
        )
        saturated = not np.array_equal(joint_target_array, unclipped_target)
        elapsed_ms = (self.monotonic() - started) * 1000.0
        diagnostics = PiperCartesianServoDiagnostics(
            state="singular" if orientation_scale < 1.0 else "nominal",
            minimum_singular_value=sigma_min,
            damping=damping,
            orientation_scale=orientation_scale,
            position_error_m=float(np.linalg.norm(position_error)),
            orientation_error_rad=float(np.linalg.norm(orientation_error)),
            maximum_joint_velocity_rad_s=float(np.max(np.abs(qdot))),
            solver_duration_ms=elapsed_ms,
            joint_limit_saturated=saturated,
        )
        return PiperCartesianServoStep(tuple(float(value) for value in joint_target_array), diagnostics)
