"""Hub-managed Piper construction without duplicating teleoperation semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Any

from lekit.control.model import CameraStreamDescriptor, NodePresentation
from lekit.control.robot import HoldResult, ObservationSink, PassiveHold, RobotNode, RobotNodeConfig
from lekit.control.runtime import Runtime
from lekit.control.video import RobotVideoServer, RobotVideoServerConfig
from lekit.teleoperators.isaac_teleop.protocol import (
    ACTION_SCHEMA,
    ACTION_SCHEMA_VERSION,
    decode_action_frame,
)
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
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
    observation_sinks: tuple[ObservationSink, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.node, RobotNodeConfig):
            raise TypeError("node must be a RobotNodeConfig")
        if not isinstance(self.robot, PiperRobotConfig):
            raise TypeError("robot must be a PiperRobotConfig")
        if not isinstance(self.processor, PiperTeleopProcessorConfig):
            raise TypeError("processor must be a PiperTeleopProcessorConfig")
        if not isinstance(self.enable_motion, bool):
            raise ValueError("enable_motion must be a bool")
        if not isinstance(self.observation_sinks, tuple) or not all(
            callable(getattr(sink, "publish", None)) for sink in self.observation_sinks
        ):
            raise TypeError("observation_sinks must be a tuple of ObservationSink")


def _decode_piper_cameras(value: Mapping[str, object]) -> dict[str, CameraConfig]:
    """Decode the only JSON camera forms supported by the Piper process.

    Importing these LeRobot config classes is intentionally side-effect free;
    hardware is opened only when the resulting :class:`PiperRobot` connects.
    """

    if not isinstance(value, Mapping):
        raise ValueError("Piper cameras must be a mapping of camera names to configurations")

    cameras: dict[str, CameraConfig] = {}
    for name, raw_config in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Piper camera names must be non-empty strings")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Piper camera {name!r} must be an object")

        config_values = dict(raw_config)
        camera_type = config_values.pop("type", None)
        if camera_type == "opencv":
            if isinstance(config_values.get("index_or_path"), str):
                config_values["index_or_path"] = Path(config_values["index_or_path"])
            config_class: type[CameraConfig] = OpenCVCameraConfig
        elif camera_type == "realsense":
            config_class = RealSenseCameraConfig
        else:
            raise ValueError(
                f"Piper camera {name!r} type must be either 'opencv' or 'realsense'"
            )
        try:
            cameras[name] = config_class(**config_values)
        except TypeError as error:
            raise ValueError(f"invalid Piper camera {name!r}: {error}") from error
    return cameras


def make_piper_video_server(
    config: PiperRobotConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    advertise_host: str | None = None,
) -> RobotVideoServer:
    """Build, but do not start, Piper's RGB-only browser video server.

    The caller owns the returned server lifecycle.  Pass it through
    ``PiperNodeConfig(observation_sinks=(server,))`` before starting the node,
    then call ``server.start()`` and ``server.stop()`` around the control loop.
    """

    if not isinstance(config, PiperRobotConfig):
        raise TypeError("config must be a PiperRobotConfig")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("video host must not be empty")
    if advertise_host is not None and (not isinstance(advertise_host, str) or not advertise_host.strip()):
        raise ValueError("advertise_host must be non-empty when present")

    bind_host = host.strip()
    public_host = advertise_host.strip().strip("[]") if advertise_host is not None else bind_host
    if bind_host in {"0.0.0.0", "::", "*"} and advertise_host is None:
        raise ValueError("advertise_host is required for a wildcard video host")
    rendered_host = f"[{public_host}]" if ":" in public_host else public_host
    base_url = f"http://{rendered_host}:{port}"
    cameras = tuple(
        CameraStreamDescriptor(
            name=name,
            stream_url=f"{base_url}/api/cameras/{name}/stream.mjpg",
            width=int(camera.width),
            height=int(camera.height),
            fps=float(camera.fps),
        )
        for name, camera in config.cameras.items()
        if getattr(camera, "use_rgb", True)
    )
    if not cameras:
        raise ValueError("Piper video requires at least one configured RGB camera")
    return RobotVideoServer(RobotVideoServerConfig(cameras=cameras, host=bind_host, port=port))


def piper_video_presentation(server: RobotVideoServer) -> NodePresentation:
    """Return the registration presentation matching a Piper video server."""

    if not isinstance(server, RobotVideoServer):
        raise TypeError("server must be a RobotVideoServer")
    cameras = server.describe()
    first_url = cameras[0].stream_url.rsplit("/api/cameras/", 1)[0]
    return NodePresentation(video_status_url=f"{first_url}/api/cameras", cameras=cameras)


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
        arm_after_release = getattr(self._pipeline.steps[0], "arm_after_validated_release", None)
        if callable(arm_after_release):
            arm_after_release()
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
    return RobotNode(
        robot,
        processor,
        node_config,
        runtime=runtime,
        hold=hold,
        observation_sinks=config.observation_sinks,
    )


__all__ = [
    "PiperIsaacPayloadProcessor",
    "PiperNodeConfig",
    "_decode_piper_cameras",
    "make_piper_video_server",
    "make_piper_robot_node",
    "piper_video_presentation",
    "piper_active_hold",
]
