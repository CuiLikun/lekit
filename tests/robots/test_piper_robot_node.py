from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lekit.control.robot import PassiveHold, RobotNodeConfig
from lekit.robots.piper import robot_node
from lekit.robots.piper.piper_robot import PiperRobot, PiperRobotConfig
from lekit.robots.piper.robot_node import (
    PiperIsaacPayloadProcessor,
    PiperNodeConfig,
    make_piper_robot_node,
    piper_active_hold,
)
from lekit.robots.piper.teleop_processor import PiperTeleopProcessorConfig
from lekit.teleoperators.isaac_teleop.protocol import TeleopFrame, encode_action_frame, neutral_action


class FakePipeline:
    def __init__(self) -> None:
        self.steps = [
            SimpleNamespace(
                config=SimpleNamespace(hand="right"),
                state=SimpleNamespace(value="idle"),
                fault_reason=None,
            )
        ]
        self.calls: list[tuple[dict[str, object], dict[str, float]]] = []
        self.reset_count = 0
        self.result = dict.fromkeys(PiperRobot._EEF_KEYS, 0.0)

    def __call__(self, value):
        action, observation = value
        self.calls.append((action, observation))
        return self.result

    def reset(self) -> None:
        self.reset_count += 1


class FakePiper:
    def __init__(self, config: PiperRobotConfig) -> None:
        self.config = config
        self.sent: list[dict[str, float]] = []
        self.send_error: Exception | None = None

    @property
    def observation_features(self):
        return dict.fromkeys(PiperRobot._EEF_KEYS, float)

    @property
    def action_features(self):
        return dict.fromkeys(PiperRobot._EEF_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return False

    def send_action(self, action):
        self.sent.append(dict(action))
        if self.send_error is not None:
            raise self.send_error
        return action


class UnusedRuntime:
    pass


def test_payload_processor_decodes_frame_and_reuses_existing_pipeline():
    pipeline = FakePipeline()
    processor = PiperIsaacPayloadProcessor(pipeline)
    action = neutral_action()
    action["right.is_tracking"] = True
    action["right.is_engaged"] = True
    frame = TeleopFrame("xr-session", 4, 10, 20, action)
    observation = {key: float(index) for index, key in enumerate(PiperRobot._EEF_KEYS)}

    assert processor(encode_action_frame(frame), observation) == pipeline.result
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0][1] == observation
    assert pipeline.calls[0][0]["right.is_tracking"] is True
    assert pipeline.calls[0][0]["right.is_engaged"] is True
    assert processor.status() == {
        "processor_state": "idle",
        "tracking": True,
        "engaged": True,
        "error": None,
    }
    processor.reset()
    assert pipeline.reset_count == 1


def test_active_hold_sends_one_complete_measured_tcp_and_reports_passive_failures():
    piper = FakePiper(PiperRobotConfig())
    observation = {key: float(index) for index, key in enumerate(PiperRobot._EEF_KEYS)}

    assert piper_active_hold(piper, observation, "watchdog").active is True
    assert piper.sent == [observation]

    incomplete = piper_active_hold(piper, {"ee.x": 0.1}, "feedback")
    assert incomplete.active is False
    assert piper.sent == [observation]

    piper.send_error = RuntimeError("sdk failed")
    failed = piper_active_hold(piper, observation, "watchdog")
    assert failed.active is False
    assert failed.detail is not None and "sdk failed" in failed.detail


def test_factory_wires_existing_processor_and_keeps_dry_run_passive(monkeypatch, tmp_path: Path):
    created: list[FakePiper] = []
    pipeline = FakePipeline()

    def make_robot(config: PiperRobotConfig) -> FakePiper:
        result = FakePiper(config)
        created.append(result)
        return result

    monkeypatch.setattr(robot_node, "PiperRobot", make_robot)
    monkeypatch.setattr(robot_node, "make_piper_isaac_processor", lambda config: pipeline)
    node_config = RobotNodeConfig(
        node_id_path=tmp_path / "piper-id",
        display_name="piper",
        accepted_payload_schemas=("lekit.isaac_teleop.action.v1",),
    )
    config = PiperNodeConfig(
        node=node_config,
        robot=PiperRobotConfig(include_gripper=False),
        processor=PiperTeleopProcessorConfig(include_gripper=True),
    )

    node = make_piper_robot_node(config, UnusedRuntime())

    assert created[0].config.auto_enable is False
    assert pipeline.steps[0].config.hand == "right"
    assert node.config.control_enabled is False
    assert isinstance(node._hold, PassiveHold)

    motion_node = make_piper_robot_node(replace(config, enable_motion=True), UnusedRuntime())
    assert motion_node.config.control_enabled is True
    assert motion_node._hold is piper_active_hold


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_active_hold_rejects_nonfinite_tcp_without_sending(value: float):
    piper = FakePiper(PiperRobotConfig())
    observation = dict.fromkeys(PiperRobot._EEF_KEYS, 0.0)
    observation["ee.y"] = value

    assert piper_active_hold(piper, observation, "feedback").active is False
    assert piper.sent == []
