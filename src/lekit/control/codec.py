"""Strict codecs for Control Hub management and action messages."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from typing import Any

from .model import (
    ActionEnvelope,
    CameraStreamDescriptor,
    ManagementMessage,
    NodeDescriptor,
    NodePresentation,
    NodeRole,
)

_HEADER_LENGTH = struct.Struct("!I")
_MAX_HEADER_BYTES = 64 * 1024
_ACTION_HEADER_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "handle_id",
        "hub_epoch",
        "fencing_token",
        "controller_id",
        "controller_session_id",
        "stream_session_id",
        "sequence",
        "captured_monotonic_ns",
        "captured_utc_ns",
        "payload_schema",
    }
)
_MANAGEMENT_FIELDS = frozenset(
    {
        "protocol_version",
        "kind",
        "correlation_id",
        "sender_id",
        "sender_session_id",
        "sequence",
        "sent_at_ns",
        "body",
    }
)
_PRESENTATION_FIELDS = frozenset({"monitor_url", "video_status_url", "cameras"})
_CAMERA_DESCRIPTOR_FIELDS = frozenset({"name", "stream_url", "width", "height", "fps"})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _decode_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = data.decode("utf-8")
        value = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def encode_management(message: ManagementMessage) -> bytes:
    """Encode a management message as one compact JSON object."""
    if not isinstance(message, ManagementMessage):
        raise ValueError("message must be a ManagementMessage")
    return json.dumps(
        {
            "protocol_version": message.protocol_version,
            "kind": message.kind,
            "correlation_id": message.correlation_id,
            "sender_id": message.sender_id,
            "sender_session_id": message.sender_session_id,
            "sequence": message.sequence,
            "sent_at_ns": message.sent_at_ns,
            "body": _thaw_json_value(message.body),
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def decode_management(data: bytes) -> ManagementMessage:
    """Decode a strict management JSON object."""
    if not isinstance(data, bytes):
        raise ValueError("management data must be bytes")
    value = _decode_object(data, label="management message")
    _require_exact_fields(value, _MANAGEMENT_FIELDS, label="management message")
    if not isinstance(value["body"], dict):
        raise ValueError("management message body must be a JSON object")
    try:
        return ManagementMessage(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid management message") from error


def decode_node_descriptor(value: Mapping[str, Any]) -> NodeDescriptor:
    """Decode the strict wire representation of one registered Node descriptor."""
    if not isinstance(value, Mapping):
        raise ValueError("registration descriptor body is invalid")
    values = dict(value)
    try:
        values["role"] = NodeRole(values["role"])
        for name in ("capabilities", "action_schemas", "control_modes"):
            values[name] = tuple(values[name])
        if "presentation" not in values:
            values["presentation"] = NodePresentation()
        else:
            presentation = values["presentation"]
            if not isinstance(presentation, Mapping):
                raise ValueError("presentation must be an object")
            _require_exact_fields(presentation, _PRESENTATION_FIELDS, label="presentation")
            cameras = presentation["cameras"]
            if not isinstance(cameras, (list, tuple)):
                raise ValueError("cameras must be an array")
            decoded_cameras = []
            for camera in cameras:
                if not isinstance(camera, Mapping):
                    raise ValueError("camera descriptor must be an object")
                _require_exact_fields(camera, _CAMERA_DESCRIPTOR_FIELDS, label="camera descriptor")
                decoded_cameras.append(CameraStreamDescriptor(**dict(camera)))
            values["presentation"] = NodePresentation(
                monitor_url=presentation["monitor_url"],
                video_status_url=presentation["video_status_url"],
                cameras=tuple(decoded_cameras),
            )
        return NodeDescriptor(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("registration descriptor body is invalid") from error


def encode_action_envelope(envelope: ActionEnvelope) -> bytes:
    """Encode an action as a length-prefixed JSON header and opaque payload bytes."""
    if not isinstance(envelope, ActionEnvelope):
        raise ValueError("envelope must be an ActionEnvelope")
    header = json.dumps(
        {
            "schema_name": "lekit.control.action",
            "schema_version": 1,
            "handle_id": envelope.handle_id,
            "hub_epoch": envelope.hub_epoch,
            "fencing_token": envelope.fencing_token,
            "controller_id": envelope.controller_id,
            "controller_session_id": envelope.controller_session_id,
            "stream_session_id": envelope.stream_session_id,
            "sequence": envelope.sequence,
            "captured_monotonic_ns": envelope.captured_monotonic_ns,
            "captured_utc_ns": envelope.captured_utc_ns,
            "payload_schema": envelope.payload_schema,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("header length is invalid")
    return _HEADER_LENGTH.pack(len(header)) + header + envelope.payload


def decode_action_envelope(data: bytes) -> ActionEnvelope:
    """Decode a single length-prefixed action envelope without interpreting its payload."""
    if not isinstance(data, bytes) or len(data) < _HEADER_LENGTH.size:
        raise ValueError("header length is invalid")
    header_length = _HEADER_LENGTH.unpack(data[: _HEADER_LENGTH.size])[0]
    if header_length > _MAX_HEADER_BYTES or len(data) < _HEADER_LENGTH.size + header_length:
        raise ValueError("header length is invalid")
    header_start = _HEADER_LENGTH.size
    header_end = header_start + header_length
    header = _decode_object(data[header_start:header_end], label="action header")
    _require_exact_fields(header, _ACTION_HEADER_FIELDS, label="action header")
    if (
        header["schema_name"] != "lekit.control.action"
        or isinstance(header["schema_version"], bool)
        or not isinstance(header["schema_version"], int)
        or header["schema_version"] != 1
    ):
        raise ValueError("action header schema is invalid")
    try:
        return ActionEnvelope(
            handle_id=header["handle_id"],
            hub_epoch=header["hub_epoch"],
            fencing_token=header["fencing_token"],
            controller_id=header["controller_id"],
            controller_session_id=header["controller_session_id"],
            stream_session_id=header["stream_session_id"],
            sequence=header["sequence"],
            captured_monotonic_ns=header["captured_monotonic_ns"],
            captured_utc_ns=header["captured_utc_ns"],
            payload_schema=header["payload_schema"],
            payload=data[header_end:],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
