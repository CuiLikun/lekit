import json
import struct

import pytest

from lekit.control import (
    ActionEnvelope,
    ControlHandle,
    HubSnapshot,
    ManagementMessage,
    NodeDescriptor,
    NodeRole,
    TimingConfig,
    decode_action_envelope,
    decode_management,
    encode_action_envelope,
    encode_management,
    load_or_create_node_id,
    normalize_feature_metadata,
)


def action_envelope() -> ActionEnvelope:
    return ActionEnvelope(
        handle_id="handle-1",
        hub_epoch="epoch-1",
        fencing_token=7,
        controller_id="quest",
        controller_session_id="controller-session",
        stream_session_id="stream-1",
        sequence=3,
        captured_monotonic_ns=11,
        captured_utc_ns=12,
        payload_schema="lekit.isaac_teleop.action.v1",
        payload=b"\x00opaque\xff",
    )


def controller_descriptor(**overrides: object) -> NodeDescriptor:
    fields = {
        "protocol_version": 1,
        "schema_version": 1,
        "node_id": "quest",
        "session_id": "controller-session",
        "role": NodeRole.CONTROLLER,
        "display_name": "Quest controller",
        "administratively_enabled": True,
        "capabilities": ("teleop",),
        "action_schemas": ("lekit.isaac_teleop.action.v1",),
        "control_modes": ("teleop",),
        "action_endpoint": "tcp://controller:5557",
        "observation_features": {"camera": {"shape": (480, 640, 3)}},
        "action_features": {"joint": {"dtype": float, "shape": (7,)}},
        "software_version": "1.0.0",
        "diagnostics": {"source": "test"},
    }
    fields.update(overrides)
    return NodeDescriptor(**fields)  # type: ignore[arg-type]


def test_action_envelope_round_trip_keeps_payload_opaque():
    envelope = action_envelope()
    assert decode_action_envelope(encode_action_envelope(envelope)) == envelope


def test_action_codec_rejects_truncated_header():
    with pytest.raises(ValueError, match="header length"):
        decode_action_envelope(b"\x00\x00\x00\x10{}")


def test_node_id_is_stable_across_loads(tmp_path):
    path = tmp_path / "node-id"
    assert load_or_create_node_id(path) == load_or_create_node_id(path)


def test_action_codec_rejects_boolean_integer_fields_and_empty_payload():
    encoded = encode_action_envelope(action_envelope())
    header_length = struct.unpack("!I", encoded[:4])[0]
    header = json.loads(encoded[4 : 4 + header_length])
    header["sequence"] = True
    invalid_integer = json.dumps(header, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="sequence"):
        decode_action_envelope(struct.pack("!I", len(invalid_integer)) + invalid_integer + b"x")

    header["sequence"] = 3
    header["schema_version"] = True
    invalid_schema_version = json.dumps(header, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="schema"):
        decode_action_envelope(struct.pack("!I", len(invalid_schema_version)) + invalid_schema_version + b"x")

    with pytest.raises(ValueError, match="payload"):
        decode_action_envelope(encoded[: -len(action_envelope().payload)])


def test_action_codec_requires_exact_header_fields_and_header_size_limit():
    encoded = encode_action_envelope(action_envelope())
    header_length = struct.unpack("!I", encoded[:4])[0]
    header = json.loads(encoded[4 : 4 + header_length])
    del header["payload_schema"]
    missing = json.dumps(header, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="fields"):
        decode_action_envelope(struct.pack("!I", len(missing)) + missing + b"x")

    with pytest.raises(ValueError, match="header length"):
        decode_action_envelope(struct.pack("!I", 65537) + b"{}")


def test_action_codec_rejects_oversized_header_during_encoding():
    oversized = ActionEnvelope(
        handle_id="handle-1",
        hub_epoch="epoch-1",
        fencing_token=7,
        controller_id="quest",
        controller_session_id="controller-session",
        stream_session_id="stream-1",
        sequence=3,
        captured_monotonic_ns=11,
        captured_utc_ns=12,
        payload_schema="x" * 66000,
        payload=b"opaque",
    )

    with pytest.raises(ValueError, match="header length"):
        encode_action_envelope(oversized)


def test_management_message_round_trip_uses_strict_compact_json():
    message = ManagementMessage(
        protocol_version=1,
        kind="heartbeat",
        correlation_id="request-1",
        sender_id="quest",
        sender_session_id="session-1",
        sequence=2,
        sent_at_ns=3,
        body={"status": "online", "nested": [1, True, None]},
    )

    encoded = encode_management(message)
    assert encoded == (
        b'{"protocol_version":1,"kind":"heartbeat","correlation_id":"request-1",'
        b'"sender_id":"quest","sender_session_id":"session-1","sequence":2,'
        b'"sent_at_ns":3,"body":{"status":"online","nested":[1,true,null]}}'
    )
    assert decode_management(encoded) == message


def test_management_codec_rejects_unknown_fields_and_non_object_body():
    message = ManagementMessage(1, "heartbeat", "c", "sender", "session", 0, 0, {})
    fields = json.loads(encode_management(message))
    fields["extra"] = "not allowed"
    with pytest.raises(ValueError, match="fields"):
        decode_management(json.dumps(fields).encode())

    fields.pop("extra")
    fields["body"] = []
    with pytest.raises(ValueError, match="body"):
        decode_management(json.dumps(fields).encode())


def test_node_descriptor_normalizes_only_json_compatible_feature_metadata():
    descriptor = controller_descriptor()
    assert descriptor.action_features["joint"]["dtype"] == "float"
    assert tuple(descriptor.action_features["joint"]["shape"]) == (7,)
    assert tuple(descriptor.observation_features["camera"]["shape"]) == (480, 640, 3)
    assert normalize_feature_metadata({"dtype": bool, "shape": (1,)}) == {
        "dtype": "bool",
        "shape": [1],
    }

    with pytest.raises(ValueError, match="feature metadata"):
        normalize_feature_metadata({"value": object()})


def test_node_descriptor_deep_freezes_feature_metadata():
    descriptor = controller_descriptor()

    with pytest.raises(AttributeError):
        descriptor.action_features["joint"]["shape"].append(8)

    assert descriptor.action_features["joint"]["shape"] == (7,)


def test_node_descriptor_deep_freezes_diagnostics():
    descriptor = controller_descriptor(diagnostics={"metrics": {"samples": [1]}})

    with pytest.raises(TypeError):
        descriptor.diagnostics["metrics"]["samples"] = (1, 2)


def test_management_message_deep_freezes_body():
    message = ManagementMessage(1, "heartbeat", "c", "sender", "session", 0, 0, {"nested": {"ids": [1]}})

    with pytest.raises(AttributeError):
        message.body["nested"]["ids"].append(2)


def test_hub_snapshot_deep_freezes_alerts():
    snapshot = HubSnapshot(
        version=0,
        hub_epoch="epoch-1",
        generated_at_ns=0,
        nodes=(),
        controls=(),
        alerts=({"details": {"codes": ["stale"]}},),
    )

    with pytest.raises(AttributeError):
        snapshot.alerts[0]["details"]["codes"].append("mismatch")


def test_node_descriptor_enforces_controller_endpoint_and_schema_requirements():
    with pytest.raises(ValueError, match="action_endpoint"):
        controller_descriptor(action_endpoint=None)
    with pytest.raises(ValueError, match="action_schemas"):
        controller_descriptor(action_schemas=())
    with pytest.raises(ValueError, match="action_endpoint"):
        controller_descriptor(action_endpoint="")


def test_control_handle_rejects_invalid_timing_or_fencing_values():
    fields = {
        "handle_id": "handle-1",
        "hub_epoch": "epoch-1",
        "robot_id": "robot",
        "robot_session_id": "robot-session",
        "controller_id": "quest",
        "controller_session_id": "controller-session",
        "controller_action_endpoint": "tcp://controller:5557",
        "action_schema": "lekit.isaac_teleop.action.v1",
        "control_mode": "teleop",
        "fencing_token": 1,
        "issued_at_ns": 1,
        "expires_at_ns": 2,
    }
    with pytest.raises(ValueError, match="invalid"):
        ControlHandle(**(fields | {"fencing_token": 0}))
    with pytest.raises(ValueError, match="invalid"):
        ControlHandle(**(fields | {"expires_at_ns": 1}))


def test_timing_config_defaults_validate_and_round_to_nanoseconds():
    timing = TimingConfig(action_stale_s=0.1000000006, heartbeat_rate_hz=3.0)
    assert timing.action_rate_hz == 60.0
    assert timing.action_stale_ns == 100000001
    assert timing.heartbeat_interval_ns == 333333333
    assert timing.renewal_interval_ns == 1000000000
    assert timing.handle_ttl_ns == 3000000000

    with pytest.raises(ValueError, match="finite positive"):
        TimingConfig(action_rate_hz=0.0)
