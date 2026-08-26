from __future__ import annotations

import json

import numpy as np
import pytest

from lekit.teleoperators.isaac_teleop.protocol import (
    ACTION_SCHEMA,
    ACTION_SCHEMA_VERSION,
    TeleopFrame,
    action_features,
    decode_action_frame,
    encode_action_frame,
    neutral_action,
)

HAND_FIELDS = (
    "translation",
    "rotation",
    "aim_translation",
    "aim_rotation",
    "squeeze",
    "trigger",
    "thumbstick",
    "thumbstick_click",
    "primary_button",
    "secondary_button",
    "menu_button",
    "is_tracking",
    "is_aim_tracking",
    "is_engaged",
)


def expected_keys() -> set[str]:
    return {f"{side}.{field}" for side in ("left", "right") for field in HAND_FIELDS}


def test_action_schema_and_neutral_action_have_the_complete_safe_contract() -> None:
    features = action_features()
    action = neutral_action()

    assert set(features) == expected_keys()
    assert set(action) == expected_keys()
    for side in ("left", "right"):
        np.testing.assert_array_equal(action[f"{side}.translation"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(action[f"{side}.rotation"], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(action[f"{side}.aim_translation"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(action[f"{side}.aim_rotation"], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(action[f"{side}.thumbstick"], [0.0, 0.0])
        assert action[f"{side}.is_tracking"] is False
        assert action[f"{side}.is_aim_tracking"] is False
        assert action[f"{side}.is_engaged"] is False


def test_action_frame_round_trip_preserves_arrays_scalars_and_metadata() -> None:
    action = neutral_action()
    action["left.translation"] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    action["right.rotation"] = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32)
    action["left.trigger"] = 0.75
    action["right.is_tracking"] = True
    frame = TeleopFrame(
        session_id="8de2a97f-7332-43a7-8153-0afe872855f1",
        sequence=42,
        captured_monotonic_ns=123_456,
        captured_utc_ns=987_654,
        action=action,
    )

    decoded = decode_action_frame(encode_action_frame(frame))

    assert decoded.session_id == frame.session_id
    assert decoded.sequence == 42
    assert decoded.captured_monotonic_ns == 123_456
    assert decoded.captured_utc_ns == 987_654
    assert decoded.action["left.translation"].dtype == np.float32
    np.testing.assert_allclose(decoded.action["left.translation"], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(decoded.action["right.rotation"], [0.0, 0.5, 0.0, 0.5])
    assert decoded.action["left.trigger"] == pytest.approx(0.75)
    assert decoded.action["right.is_tracking"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema="wrong"), "schema"),
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(schema_version=True), "schema_version"),
        (lambda payload: payload.update(sequence=-1), "sequence"),
        (lambda payload: payload.update(unexpected="field"), "top-level fields"),
        (lambda payload: payload["action"].pop("left.trigger"), "action fields"),
        (lambda payload: payload["action"].update({"left.trigger": float("nan")}), "finite"),
        (lambda payload: payload["action"].update({"left.translation": [1.0, 2.0]}), "shape"),
        (
            lambda payload: payload["action"].update({"left.translation": ["1.0", 2.0, 3.0]}),
            "numeric values",
        ),
        (
            lambda payload: payload["action"].update({"left.translation": [True, 2.0, 3.0]}),
            "numeric values",
        ),
    ],
)
def test_decoder_rejects_incompatible_or_unsafe_payloads(mutation, message: str) -> None:
    payload = {
        "schema": ACTION_SCHEMA,
        "schema_version": ACTION_SCHEMA_VERSION,
        "session_id": "session-1",
        "sequence": 0,
        "captured_monotonic_ns": 1,
        "captured_utc_ns": 2,
        "action": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in neutral_action().items()
        },
    }
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        decode_action_frame(json.dumps(payload, allow_nan=True).encode())


def test_encoder_rejects_extra_action_fields() -> None:
    action = neutral_action()
    action["robot.command"] = 1.0
    frame = TeleopFrame("session-1", 0, 1, 2, action)

    with pytest.raises(ValueError, match="action fields"):
        encode_action_frame(frame)
