from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def infer_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    return importlib.import_module("examples.isaac_teleop_to_jaka.infer")


@pytest.mark.parametrize(
    ("control_mode", "expected_action"),
    [
        (
            "eef",
            {
                "ee.x": 7.0,
                "ee.y": 8.0,
                "ee.z": 9.0,
                "ee.roll": 10.0,
                "ee.pitch": 11.0,
                "ee.yaw": 12.0,
                "gripper.pos": 6.0,
            },
        ),
        (
            "joint",
            {
                "joint_1.pos": 0.0,
                "joint_2.pos": 1.0,
                "joint_3.pos": 2.0,
                "joint_4.pos": 3.0,
                "joint_5.pos": 4.0,
                "joint_6.pos": 5.0,
                "gripper.pos": 6.0,
            },
        ),
    ],
)
def test_policy_vector_uses_selected_arm_representation(infer_module, control_mode, expected_action):
    names = list(infer_module._POLICY_ACTION_NAMES)
    action, vector = infer_module._to_robot_action(
        torch.tensor(range(13), dtype=torch.float32), names, control_mode=control_mode
    )

    assert vector.tolist() == list(range(13))
    assert action == expected_action


def test_parser_receives_dataclass_config_type(infer_module):
    annotations = inspect.getfullargspec(infer_module.infer.__wrapped__).annotations

    assert annotations["cfg"] is infer_module.InferConfig


def test_proxy_config_accepts_explicit_zmq_address(infer_module):
    config = infer_module.ProxyConfig(addr="tcp://127.0.0.1:9000")

    assert config.addr == "tcp://127.0.0.1:9000"

    with pytest.raises(ValueError, match="tcp://host:port"):
        infer_module.ProxyConfig(addr="127.0.0.1:9000")


@pytest.mark.parametrize(
    ("control_mode", "expected_servo_process", "expected_limit", "expected_filter"),
    [("eef", True, 0.05, "cartesian_nlf"), ("joint", False, None, "joint_nlf")],
)
def test_inference_configures_servo_for_selected_representation(
    infer_module, monkeypatch, control_mode, expected_servo_process, expected_limit, expected_filter
):
    class FakeRobot:
        def __init__(self, config=None):
            self.config = config

        def connect(self):
            pass

    config = SimpleNamespace(
        auto_enable_servo=True,
        separate_feedback_connection=True,
        servo_process=True,
        servo_filter_mode="none",
        max_relative_target=0.05,
        user_frame_id=0,
    )
    monkeypatch.setattr(infer_module, "JakaRobot", FakeRobot)
    monkeypatch.setattr(infer_module, "make_robot_from_config", lambda config: FakeRobot(config))

    infer_module._configure_robot(config, control_mode=control_mode)

    assert config.auto_enable_servo is False
    assert config.separate_feedback_connection is False
    assert config.servo_process is expected_servo_process
    assert config.max_relative_target == expected_limit
    assert config.servo_filter_mode == expected_filter


def test_reset_policy_reconnects_to_clear_queued_actions(infer_module):
    addresses = []

    class FakeProxy:
        def switch_policy(self, address):
            addresses.append(address)

    infer_module._reset_policy(FakeProxy(), "tcp://127.0.0.1:9000")

    assert addresses == ["tcp://127.0.0.1:9000"]


def test_policy_gripper_gate_only_emits_state_transitions(infer_module):
    gate = infer_module.PolicyGripperGate()

    assert gate.next_command(0.9, observed_position=0.0, now=0.0) == 1.0
    gate.mark_applied(1.0)
    assert gate.next_command(0.9, observed_position=0.0, now=0.1) is None
    assert gate.next_command(0.5, observed_position=0.0, now=0.2) is None
    assert gate.next_command(0.1, observed_position=0.0, now=0.3) == 0.0


def test_policy_gripper_gate_retries_a_failed_transition_at_a_limited_rate(infer_module):
    gate = infer_module.PolicyGripperGate(retry_interval_s=1.0)

    assert gate.next_command(0.9, observed_position=0.0, now=0.0) == 1.0
    gate.mark_failed(now=0.0)
    assert gate.next_command(0.9, observed_position=0.0, now=0.5) is None
    assert gate.next_command(0.9, observed_position=0.0, now=1.0) == 1.0


def test_manual_gripper_toggle_does_not_lock_out_policy_commands(infer_module):
    gate = infer_module.PolicyGripperGate()

    assert gate.toggle_manual(observed_position=0.0) == 1.0
    gate.mark_applied(1.0)
    assert gate.next_command(0.0, observed_position=1.0, now=0.0) == 0.0
    gate.mark_applied(0.0)
    assert gate.toggle_manual(observed_position=1.0) == 1.0


def test_policy_vector_rejects_schema_and_numeric_errors(infer_module):
    with pytest.raises(ValueError, match="schema has 2 fields"):
        infer_module._to_robot_action(torch.zeros(3), ["ee.x", "ee.y"], control_mode="eef")
    with pytest.raises(ValueError, match="non-finite"):
        infer_module._to_robot_action(
            torch.tensor([float("nan")] + [0.0] * 12),
            list(infer_module._POLICY_ACTION_NAMES),
            control_mode="eef",
        )


def test_rerun_frame_contains_state_action_policy_and_images(infer_module):
    captured = []

    class FakeLogger:
        def log(self, frame):
            captured.append(frame)

    class FakeCamera:
        use_rgb = True

    class FakeRobot:
        action_features = ["joint_1.pos", "gripper.pos", "ee.x"]
        cameras = {"hand": FakeCamera()}

    observation = {
        "joint_1.pos": 0.1,
        "gripper.pos": 0.2,
        "ee.x": 0.3,
        "hand": np.zeros((2, 3, 3), dtype=np.uint8),
    }
    infer_module._log_rerun(
        FakeLogger(),
        FakeRobot(),
        observation,
        {"ee.x": 0.4},
        np.array([0.4]),
        task="pick tube",
        state="recording",
        metrics={"loop_rate_hz": 30.0},
    )

    assert len(captured) == 1
    frame = captured[0]
    assert frame["observation.state"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert frame["action.ee.x"] == pytest.approx(0.4)
    assert frame["policy"].tolist() == pytest.approx([0.4])
    assert frame["metrics.loop_rate_hz"] == pytest.approx(30.0)
    assert frame["observation.images.hand"].shape == (2, 3, 3)
