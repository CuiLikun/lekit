"""Versioned, robot-independent wire protocol for Isaac controller snapshots."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from lerobot.types import RobotAction

ACTION_SCHEMA = "lekit.isaac_teleop.action"
ACTION_SCHEMA_VERSION = 1
CONTROLLER_SIDES = ("left", "right")

_FRAME_KEYS = {
    "schema",
    "schema_version",
    "session_id",
    "sequence",
    "captured_monotonic_ns",
    "captured_utc_ns",
    "action",
}

_ARRAY_FIELDS = {
    "translation": 3,
    "rotation": 4,
    "aim_translation": 3,
    "aim_rotation": 4,
    "thumbstick": 2,
}
_FLOAT_FIELDS = (
    "squeeze",
    "trigger",
    "thumbstick_click",
    "primary_button",
    "secondary_button",
    "menu_button",
)
_BOOL_FIELDS = ("is_tracking", "is_aim_tracking", "is_engaged")
_HAND_FIELDS = (*_ARRAY_FIELDS, *_FLOAT_FIELDS, *_BOOL_FIELDS)
ACTION_KEYS = tuple(f"{side}.{field}" for side in CONTROLLER_SIDES for field in _HAND_FIELDS)


def action_features() -> dict[str, dict[str, Any]]:
    """Return the stable LeRobot feature declaration shared by producer and subscribers."""

    hand_features = {
        "translation": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"x_right": 0, "y_forward": 1, "z_up": 2},
        },
        "rotation": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"qx": 0, "qy": 1, "qz": 2, "qw": 3},
        },
        "aim_translation": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"x_right": 0, "y_forward": 1, "z_up": 2},
        },
        "aim_rotation": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"qx": 0, "qy": 1, "qz": 2, "qw": 3},
        },
        "squeeze": {"dtype": "float32", "shape": ()},
        "trigger": {"dtype": "float32", "shape": ()},
        "thumbstick": {"dtype": "float32", "shape": (2,), "names": {"x": 0, "y": 1}},
        "thumbstick_click": {"dtype": "float32", "shape": ()},
        "primary_button": {"dtype": "float32", "shape": ()},
        "secondary_button": {"dtype": "float32", "shape": ()},
        "menu_button": {"dtype": "float32", "shape": ()},
        "is_tracking": {"dtype": "bool", "shape": ()},
        "is_aim_tracking": {"dtype": "bool", "shape": ()},
        "is_engaged": {"dtype": "bool", "shape": ()},
    }
    return {
        f"{side}.{field}": dict(feature)
        for side in CONTROLLER_SIDES
        for field, feature in hand_features.items()
    }


def neutral_action() -> RobotAction:
    """Return a complete fail-safe action with both controllers disengaged."""

    action: RobotAction = {}
    for side in CONTROLLER_SIDES:
        action.update(
            {
                f"{side}.translation": np.zeros(3, dtype=np.float32),
                f"{side}.rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                f"{side}.aim_translation": np.zeros(3, dtype=np.float32),
                f"{side}.aim_rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                f"{side}.squeeze": 0.0,
                f"{side}.trigger": 0.0,
                f"{side}.thumbstick": np.zeros(2, dtype=np.float32),
                f"{side}.thumbstick_click": 0.0,
                f"{side}.primary_button": 0.0,
                f"{side}.secondary_button": 0.0,
                f"{side}.menu_button": 0.0,
                f"{side}.is_tracking": False,
                f"{side}.is_aim_tracking": False,
                f"{side}.is_engaged": False,
            }
        )
    return action


@dataclass(frozen=True)
class TeleopFrame:
    """One atomic dual-controller sample."""

    session_id: str
    sequence: int
    captured_monotonic_ns: int
    captured_utc_ns: int
    action: RobotAction


def normalize_action(action: Mapping[str, Any]) -> RobotAction:
    """Validate a complete action and return stable NumPy/Python value types."""

    expected = set(ACTION_KEYS)
    actual = set(action)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"action fields do not match schema (missing={missing}, extra={extra})")

    normalized: RobotAction = {}
    for side in CONTROLLER_SIDES:
        for field, size in _ARRAY_FIELDS.items():
            key = f"{side}.{field}"
            value = np.asarray(action[key], dtype=np.float32)
            if value.shape != (size,):
                raise ValueError(f"{key} has shape {value.shape}; expected {(size,)}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{key} must contain only finite values")
            normalized[key] = value.copy()
        for field in _FLOAT_FIELDS:
            key = f"{side}.{field}"
            value = action[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{key} must be a finite float")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{key} must be finite")
            normalized[key] = numeric
        for field in _BOOL_FIELDS:
            key = f"{side}.{field}"
            value = action[key]
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be a boolean")
            normalized[key] = bool(value)
    return normalized


def encode_action_frame(frame: TeleopFrame) -> bytes:
    """Validate and serialize an action frame as compact UTF-8 JSON."""

    normalized = normalize_action(frame.action)
    _validate_metadata(frame.session_id, frame.sequence, frame.captured_monotonic_ns, frame.captured_utc_ns)
    payload = {
        "schema": ACTION_SCHEMA,
        "schema_version": ACTION_SCHEMA_VERSION,
        "session_id": frame.session_id,
        "sequence": frame.sequence,
        "captured_monotonic_ns": frame.captured_monotonic_ns,
        "captured_utc_ns": frame.captured_utc_ns,
        "action": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in normalized.items()
        },
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_action_frame(payload: bytes) -> TeleopFrame:
    """Decode and validate one version-compatible action frame."""

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("action frame is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError("action frame must be a JSON object")
    if set(document) != _FRAME_KEYS:
        missing = sorted(_FRAME_KEYS - set(document))
        extra = sorted(set(document) - _FRAME_KEYS)
        raise ValueError(f"top-level fields do not match schema (missing={missing}, extra={extra})")
    if document.get("schema") != ACTION_SCHEMA:
        raise ValueError(f"unsupported action schema: {document.get('schema')!r}")
    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != ACTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {document.get('schema_version')!r}")
    try:
        session_id = document["session_id"]
        sequence = document["sequence"]
        captured_monotonic_ns = document["captured_monotonic_ns"]
        captured_utc_ns = document["captured_utc_ns"]
        action = document["action"]
    except KeyError as error:
        raise ValueError(f"action frame is missing {error.args[0]!r}") from error
    _validate_metadata(session_id, sequence, captured_monotonic_ns, captured_utc_ns)
    if not isinstance(action, dict):
        raise ValueError("action must be a JSON object")
    _validate_wire_action(action)
    return TeleopFrame(
        session_id=session_id,
        sequence=sequence,
        captured_monotonic_ns=captured_monotonic_ns,
        captured_utc_ns=captured_utc_ns,
        action=normalize_action(action),
    )


def _validate_wire_action(action: Mapping[str, Any]) -> None:
    """Reject JSON coercions before converting wire arrays to NumPy."""

    expected = set(ACTION_KEYS)
    actual = set(action)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"action fields do not match schema (missing={missing}, extra={extra})")

    for side in CONTROLLER_SIDES:
        for field, size in _ARRAY_FIELDS.items():
            key = f"{side}.{field}"
            value = action.get(key)
            if not isinstance(value, list) or len(value) != size:
                # Shape details are reported by ``normalize_action`` when the
                # value is a JSON array with the wrong length.
                if isinstance(value, list):
                    continue
                raise ValueError(f"{key} must be an array of numeric values")
            if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
                raise ValueError(f"{key} must contain only numeric values")


def _validate_metadata(session_id: Any, sequence: Any, monotonic_ns: Any, utc_ns: Any) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    for name, value in (
        ("sequence", sequence),
        ("captured_monotonic_ns", monotonic_ns),
        ("captured_utc_ns", utc_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "ACTION_KEYS",
    "ACTION_SCHEMA",
    "ACTION_SCHEMA_VERSION",
    "CONTROLLER_SIDES",
    "TeleopFrame",
    "action_features",
    "decode_action_frame",
    "encode_action_frame",
    "neutral_action",
    "normalize_action",
]
