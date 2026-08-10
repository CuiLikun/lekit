"""CSV diagnostics for the XR-to-JAKA Cartesian control loop."""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

_EEF_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
_EEF_KEYS = tuple(f"ee.{axis}" for axis in _EEF_AXES)

_TRACE_FIELDS = (
    "time_s",
    "phase",
    "frame_index",
    "action_source",
    "clutch_engaged",
    "clutch_released",
    "squeeze",
    "trigger",
    "frame_ms",
    "control_rate_hz",
    "control_target_hz",
    *(f"grip_{axis}_m" for axis in ("x", "y", "z")),
    *(f"raw_grip_{axis}_m" for axis in ("x", "y", "z")),
    "head_is_tracking",
    *(f"head_quat_{axis}" for axis in ("x", "y", "z", "w")),
    "control_yaw_deg",
    *(f"actual_{axis}" for axis in _EEF_AXES),
    *(f"requested_{axis}" for axis in _EEF_AXES),
    *(f"applied_{axis}" for axis in _EEF_AXES),
    *(f"tracking_error_{axis}" for axis in _EEF_AXES),
    "tracking_error_norm_m",
    "tracking_error_angle_rad",
    *(f"target_step_{axis}" for axis in _EEF_AXES),
    "target_step_norm_m",
    "target_step_angle_rad",
    "servo_active",
    "servo_worker_alive",
    "servo_representation",
    "servo_filter_mode",
    "servo_target_age_s",
    "servo_send_rate_hz",
    "servo_period_p95_ms",
    "servo_period_max_ms",
    "servo_frames_sent",
    "servo_overruns",
    "servo_queue_depth",
    *(f"servo_target_{axis}" for axis in _EEF_AXES),
    *(f"servo_commanded_{axis}" for axis in _EEF_AXES),
)


def _number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping_pose(values: Mapping[str, Any]) -> tuple[float | None, ...]:
    return tuple(_number(values.get(key)) for key in _EEF_KEYS)


def _sequence(values: object, length: int) -> tuple[float | None, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != length:
        return (None,) * length
    return tuple(_number(value) for value in values)


def _sequence_pose(values: object) -> tuple[float | None, ...]:
    return _sequence(values, 6)


def _pose_delta(
    current: tuple[float | None, ...],
    reference: tuple[float | None, ...] | None,
) -> tuple[float | None, ...]:
    if reference is None:
        return (None,) * 6
    return tuple(
        current_value - reference_value
        if current_value is not None and reference_value is not None
        else None
        for current_value, reference_value in zip(current, reference, strict=True)
    )


def _translation_norm(delta: tuple[float | None, ...]) -> float | None:
    if any(value is None for value in delta[:3]):
        return None
    return math.sqrt(sum(float(value) ** 2 for value in delta[:3]))


def _angle_norm(delta: tuple[float | None, ...]) -> float | None:
    if any(value is None for value in delta[3:]):
        return None
    wrapped = [((float(value) + math.pi) % (2.0 * math.pi)) - math.pi for value in delta[3:]]
    return math.sqrt(sum(value**2 for value in wrapped))


def _put_pose(row: dict[str, object], prefix: str, pose: tuple[float | None, ...]) -> None:
    row.update({f"{prefix}_{axis}": value for axis, value in zip(_EEF_AXES, pose, strict=True)})


class ControlTraceWriter:
    """Write one synchronized diagnostic row per foreground control frame."""

    def __init__(self, path: Path, *, flush_every: int = 30):
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.path = Path(path)
        self.flush_every = flush_every
        self._stream: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._started_at = 0.0
        self._frame_index = 0
        self._previous_phase: str | None = None
        self._previous_applied: tuple[float | None, ...] | None = None

    def __enter__(self) -> ControlTraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=_TRACE_FIELDS)
        self._writer.writeheader()
        self._stream.flush()
        self._started_at = time.monotonic()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._writer = None

    def write_frame(
        self,
        *,
        phase: str,
        raw_action: Mapping[str, Any] | None,
        action: Mapping[str, Any],
        sent_action: Mapping[str, Any],
        observation: Mapping[str, Any],
        telemetry: Mapping[str, Any],
        servo_status: Mapping[str, Any],
        frame_ms: float,
        control_rate_hz: float | None,
        control_target_hz: float,
    ) -> None:
        if self._writer is None or self._stream is None:
            raise RuntimeError("ControlTraceWriter must be entered before writing")

        actual = _mapping_pose(observation)
        requested = _mapping_pose(action)
        applied = _mapping_pose(sent_action)
        if phase != self._previous_phase:
            self._previous_applied = None
        tracking_error = _pose_delta(applied, actual)
        target_step = _pose_delta(applied, self._previous_applied)
        grip_pos = _sequence(telemetry.get("grip_pos"), 3)
        raw_grip_pos = _sequence(telemetry.get("raw_grip_pos"), 3)
        head_quat = _sequence(telemetry.get("head_quat"), 4)

        row: dict[str, object] = {
            "time_s": time.monotonic() - self._started_at,
            "phase": phase,
            "frame_index": self._frame_index,
            "action_source": "xr" if raw_action is not None else "hold",
            "clutch_engaged": bool(telemetry.get("clutch_engaged", False)),
            "clutch_released": bool(telemetry.get("clutch_released", False)),
            "squeeze": _number(telemetry.get("squeeze")),
            "trigger": _number(telemetry.get("trigger")),
            "frame_ms": frame_ms,
            "control_rate_hz": control_rate_hz,
            "control_target_hz": control_target_hz,
            "grip_x_m": grip_pos[0],
            "grip_y_m": grip_pos[1],
            "grip_z_m": grip_pos[2],
            "raw_grip_x_m": raw_grip_pos[0],
            "raw_grip_y_m": raw_grip_pos[1],
            "raw_grip_z_m": raw_grip_pos[2],
            "head_is_tracking": telemetry.get("head_is_tracking"),
            "head_quat_x": head_quat[0],
            "head_quat_y": head_quat[1],
            "head_quat_z": head_quat[2],
            "head_quat_w": head_quat[3],
            "control_yaw_deg": _number(telemetry.get("control_yaw_deg")),
            "tracking_error_norm_m": _translation_norm(tracking_error),
            "tracking_error_angle_rad": _angle_norm(tracking_error),
            "target_step_norm_m": _translation_norm(target_step),
            "target_step_angle_rad": _angle_norm(target_step),
            "servo_active": servo_status.get("active"),
            "servo_worker_alive": servo_status.get("worker_alive"),
            "servo_representation": servo_status.get("representation"),
            "servo_filter_mode": servo_status.get("filter_mode"),
            "servo_target_age_s": servo_status.get("target_age_s"),
            "servo_send_rate_hz": servo_status.get("send_rate_hz"),
            "servo_period_p95_ms": servo_status.get("period_p95_ms"),
            "servo_period_max_ms": servo_status.get("period_max_ms"),
            "servo_frames_sent": servo_status.get("frames_sent"),
            "servo_overruns": servo_status.get("overruns"),
            "servo_queue_depth": servo_status.get("queue_depth"),
        }
        _put_pose(row, "actual", actual)
        _put_pose(row, "requested", requested)
        _put_pose(row, "applied", applied)
        _put_pose(row, "tracking_error", tracking_error)
        _put_pose(row, "target_step", target_step)
        _put_pose(row, "servo_target", _sequence_pose(servo_status.get("target")))
        _put_pose(row, "servo_commanded", _sequence_pose(servo_status.get("commanded_position")))
        self._writer.writerow(row)

        self._frame_index += 1
        self._previous_phase = phase
        self._previous_applied = applied
        if self._frame_index % self.flush_every == 0:
            self._stream.flush()


__all__ = ["ControlTraceWriter"]
