"""Robot-independent relative-pose math for XR controller clutching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray | None:
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-6 else None


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    normalized = _normalize_quaternion(quaternion)
    if normalized is None:
        raise ValueError("quaternion must contain four finite, nonzero values")
    x, y, z, w = normalized
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            quaternion = np.array(
                [
                    0.25 * scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[2, 1] - m[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            quaternion = np.array(
                [
                    (m[0, 1] + m[1, 0]) / scale,
                    0.25 * scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    (m[0, 2] - m[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            quaternion = np.array(
                [
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    0.25 * scale,
                    (m[1, 0] - m[0, 1]) / scale,
                ]
            )
    normalized = _normalize_quaternion(quaternion)
    if normalized is None:
        raise ValueError("rotation matrix cannot be represented as a quaternion")
    return normalized.astype(np.float32)


def operator_frame_from_head_quaternion(head_quaternion: np.ndarray) -> np.ndarray | None:
    """Return the operator-to-OpenXR rotation from a head pose.

    The output frame is +X right, +Y forward, +Z up. It is meant to be
    sampled once at clutch engagement, so turning the head while controlling
    cannot move a stationary controller.
    """

    normalized = _normalize_quaternion(head_quaternion)
    if normalized is None:
        return None
    anchor_rotation_head = _quaternion_to_matrix(normalized)
    forward = anchor_rotation_head @ np.array([0.0, 0.0, -1.0])
    forward[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-3:
        return None
    forward /= forward_norm
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    return np.column_stack((right, forward, up))


def default_operator_frame() -> np.ndarray:
    """Return the fixed OpenXR-to-operator convention without head tracking."""

    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


@dataclass(frozen=True)
class RelativePose:
    """Relative controller pose expressed in the operator coordinate frame."""

    translation: np.ndarray
    rotation: np.ndarray
    engaged: bool


class RelativePoseClutch:
    """Latch an XR pose and return its cumulative relative transform.

    The module has no robot feedback input. A rising squeeze edge establishes
    the local zero pose, and release returns a neutral pose immediately.
    """

    def __init__(self, *, engage_threshold: float, release_threshold: float):
        self.engage_threshold = engage_threshold
        self.release_threshold = release_threshold
        self._origin_position: np.ndarray | None = None
        self._origin_rotation: np.ndarray | None = None
        self._operator_to_anchor: np.ndarray | None = None
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def reset(self) -> None:
        self._origin_position = None
        self._origin_rotation = None
        self._operator_to_anchor = None
        self._engaged = False

    def update(
        self,
        *,
        position: np.ndarray,
        quaternion: np.ndarray,
        squeeze: float,
        operator_to_anchor: np.ndarray | None,
        tracked: bool,
    ) -> RelativePose:
        """Return cumulative displacement and orientation from the engage pose."""

        normalized = _normalize_quaternion(quaternion)
        position_value = np.asarray(position, dtype=float)
        valid = (
            tracked
            and position_value.shape == (3,)
            and np.all(np.isfinite(position_value))
            and normalized is not None
            and np.isfinite(squeeze)
        )
        if not valid:
            self.reset()
            return self._neutral()

        if not self._engaged:
            if squeeze < self.engage_threshold or operator_to_anchor is None:
                return self._neutral()
            frame = np.asarray(operator_to_anchor, dtype=float)
            if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
                return self._neutral()
            self._origin_position = position_value.copy()
            self._origin_rotation = _quaternion_to_matrix(normalized)
            self._operator_to_anchor = frame.copy()
            self._engaged = True
            return self._neutral(engaged=True)

        if squeeze < self.release_threshold:
            self.reset()
            return self._neutral()

        assert self._origin_position is not None
        assert self._origin_rotation is not None
        assert self._operator_to_anchor is not None
        anchor_to_operator = self._operator_to_anchor.T
        translation = anchor_to_operator @ (position_value - self._origin_position)
        relative_anchor_rotation = _quaternion_to_matrix(normalized) @ self._origin_rotation.T
        relative_operator_rotation = anchor_to_operator @ relative_anchor_rotation @ self._operator_to_anchor
        return RelativePose(
            translation=translation.astype(np.float32),
            rotation=_matrix_to_quaternion(relative_operator_rotation),
            engaged=True,
        )

    @staticmethod
    def _neutral(*, engaged: bool = False) -> RelativePose:
        return RelativePose(
            translation=np.zeros(3, dtype=np.float32),
            rotation=_IDENTITY_QUATERNION.copy(),
            engaged=engaged,
        )
