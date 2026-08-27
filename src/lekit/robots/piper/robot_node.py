"""Hub-managed Piper construction without duplicating teleoperation semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any

from lekit.control.robot import HoldResult, PassiveHold, RobotNode, RobotNodeConfig
from lekit.control.runtime import Runtime
from lekit.teleoperators.isaac_teleop.protocol import (
    ACTION_SCHEMA,
    ACTION_SCHEMA_VERSION,
    decode_action_frame,
)
from lerobot.processor import RobotProcessorPipeline
from lerobot.types import RobotAction, RobotObservation

from .piper_robot import PiperRobot, PiperRobotConfig
from .teleop_processor import PiperTeleopProcessorConfig, make_piper_isaac_processor


@dataclass(kw_only=True)
class PiperNodeConfig:
    """Configuration for a Piper wrapped by the generic Hub ``RobotNode``."""

    node: RobotNodeConfig
    robot: PiperRobotConfig = field(default_factory=PiperRobotConfig)
    processor: PiperTeleopProcessorConfig = field(default_factory=PiperTeleopProcessorConfig)
    enable_motion: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.node, RobotNodeConfig):
            raise TypeError("node must be a RobotNodeConfig")
        if not isinstance(self.robot, PiperRobotConfig):
            raise TypeError("robot must be a PiperRobotConfig")
        if not isinstance(self.processor, PiperTeleopProcessorConfig):
            raise TypeError("processor must be a PiperTeleopProcessorConfig")
        if not isinstance(self.enable_motion, bool):
            raise ValueError("enable_motion must be a bool")


class PiperIsaacPayloadProcessor:
    """Decode an Isaac frame and delegate all retargeting to Piper's existing pipeline."""

    accepted_payload_schemas = frozenset({f"{ACTION_SCHEMA}.v{ACTION_SCHEMA_VERSION}"})

    def __init__(self, pipeline: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]):
        self._pipeline = pipeline
        step = pipeline.steps[0]
        self._hand = step.config.hand
        self._tracking = False
        self._engaged = False

    def __call__(self, payload: bytes, observation: RobotObservation) -> RobotAction:
        frame = decode_action_frame(payload)
        self._tracking = bool(frame.action[f"{self._hand}.is_tracking"])
        self._engaged = bool(frame.action[f"{self._hand}.is_engaged"])
        return self._pipeline((frame.action, observation))

    def reset(self) -> None:
        self._pipeline.reset()
        self._tracking = False
        self._engaged = False

    def status(self) -> Mapping[str, Any]:
        step = self._pipeline.steps[0]
        return {
            "processor_state": getattr(getattr(step, "state", None), "value", "unknown"),
            "tracking": self._tracking,
            "engaged": self._engaged,
            "error": getattr(step, "fault_reason", None),
        }


def piper_active_hold(
    robot: PiperRobot,
    observation: RobotObservation | None,
    reason: str,
) -> HoldResult:
    """Command one measured TCP target when feedback is complete and trustworthy."""

    if not isinstance(observation, Mapping):
        return HoldResult(active=False, detail=f"{reason}: Piper TCP feedback is unavailable")

    action: RobotAction = {}
    for key in PiperRobot._EEF_KEYS:
        value = observation.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            return HoldResult(
                active=False, detail=f"{reason}: Piper TCP feedback is incomplete or non-finite"
            )
        action[key] = float(value)

    try:
        robot.send_action(action)
    except Exception as error:  # nosec B110 - callback must report SDK safety failure to RobotNode
        return HoldResult(active=False, detail=f"{reason}: Piper active hold failed: {error}")
    return HoldResult(active=True)


def make_piper_robot_node(config: PiperNodeConfig, runtime: Runtime) -> RobotNode:
    """Build a safe Piper ``RobotNode`` with the established Isaac processor."""

    if not isinstance(config, PiperNodeConfig):
        raise TypeError("config must be a PiperNodeConfig")

    robot_config = replace(
        config.robot,
        auto_enable=config.robot.auto_enable if config.enable_motion else False,
    )
    robot = PiperRobot(robot_config)
    processor_config = replace(
        config.processor,
        include_gripper=config.processor.include_gripper and robot_config.include_gripper,
    )
    processor = PiperIsaacPayloadProcessor(make_piper_isaac_processor(processor_config))
    node_config = replace(config.node, control_enabled=config.node.control_enabled and config.enable_motion)
    hold = piper_active_hold if config.enable_motion else PassiveHold()
    return RobotNode(robot, processor, node_config, runtime=runtime, hold=hold)


__all__ = [
    "PiperIsaacPayloadProcessor",
    "PiperNodeConfig",
    "make_piper_robot_node",
    "piper_active_hold",
]
