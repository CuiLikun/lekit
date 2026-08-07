"""Pose helpers for JAKA's XYZ roll/pitch/yaw Cartesian convention."""

from __future__ import annotations

import math

import numpy as np


def _nearest_angle(angle: float, reference: float) -> float:
    """Return the 2-pi-equivalent ``angle`` nearest ``reference``."""
    return reference + (angle - reference + math.pi) % (2.0 * math.pi) - math.pi


def matrix_to_rpy(matrix: np.ndarray, *, reference_rpy: np.ndarray | None = None) -> np.ndarray:
    """Convert a matrix to a continuous XYZ roll/pitch/yaw Euler representation.

    A rotation has several equivalent Euler triples. Selecting only the principal
    value produces apparent 180-degree roll/yaw jumps near pitch +/- pi/2; when
    available, the prior command selects the nearest equivalent triple instead.
    """
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {rotation.shape}")

    reference = None if reference_rpy is None else np.asarray(reference_rpy, dtype=float)
    if reference is not None and (reference.shape != (3,) or not np.all(np.isfinite(reference))):
        raise ValueError("reference_rpy must contain three finite values")

    r00, r01 = rotation[0, 0], rotation[0, 1]
    r11 = rotation[1, 1]
    r20, r21, r22 = rotation[2, 0], rotation[2, 1], rotation[2, 2]
    pitch = float(np.arcsin(np.clip(-r20, -1.0, 1.0)))

    if abs(r20) >= 1.0 - 1e-9:
        # At gimbal lock only one roll/yaw combination is observable. Keep the
        # prior roll and solve yaw from that combination rather than forcing roll=0.
        coupled = float(np.arctan2(-r01, r11))
        roll = 0.0 if reference is None else float(reference[0])
        yaw = coupled + roll if pitch > 0.0 else coupled - roll
        result = np.array([roll, math.copysign(math.pi / 2.0, pitch), yaw])
        if reference is not None:
            result[0] = _nearest_angle(result[0], float(reference[0]))
            result[2] = _nearest_angle(result[2], float(reference[2]))
        return result

    principal = np.array(
        [
            float(np.arctan2(r21, r22)),
            pitch,
            float(np.arctan2(rotation[1, 0], r00)),
        ]
    )
    if reference is None:
        return principal

    alternate = np.array([principal[0] + math.pi, math.pi - principal[1], principal[2] + math.pi])
    candidates = np.vstack(
        [
            np.array([_nearest_angle(value, ref) for value, ref in zip(principal, reference, strict=True)]),
            np.array([_nearest_angle(value, ref) for value, ref in zip(alternate, reference, strict=True)]),
        ]
    )
    return candidates[np.argmin(np.sum((candidates - reference) ** 2, axis=1))]
