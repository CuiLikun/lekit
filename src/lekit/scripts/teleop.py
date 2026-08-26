"""Run Piper Cartesian teleoperation from an independent Isaac teleop node."""

import logging
import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from scipy.spatial.transform import Rotation

from lekit.robots.piper import (
    PiperCameraTimeoutError,
    PiperRobot,
    PiperRobotConfig,
    PiperTeleopProcessorConfig,
    make_piper_isaac_processor,
)
from lekit.teleoperators.isaac_teleop.subscriber import (
    IsaacTeleopNodeConfig,
    IsaacTeleopNodeSubscriber,
)
from lerobot.configs import parser
from lerobot.processor import RobotProcessorPipeline
from lerobot.types import RobotObservation
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

_EEF_KEYS = PiperRobot._EEF_KEYS


def _safe_piper_robot_config() -> PiperRobotConfig:
    return PiperRobotConfig(
        include_gripper=True,
        speed_percent=10,
        gripper_force_n=1.0,
        gripper_min_width_m=0.0,
        gripper_max_width_m=0.05,
        max_eef_target_lead_m=0.005,
        max_eef_target_lead_rad=math.radians(1.0),
    )


def _safe_isaac_node_config() -> IsaacTeleopNodeConfig:
    return IsaacTeleopNodeConfig(
        endpoint="tcp://127.0.0.1:5557",
        first_frame_timeout_s=5.0,
        stale_after_s=0.25,
        rearm_squeeze_threshold=0.3,
    )


def _safe_piper_processor_config() -> PiperTeleopProcessorConfig:
    return PiperTeleopProcessorConfig(
        include_gripper=True,
        translation_scale=1.0,
        rotation_scale=1.0,
        max_translation_from_anchor_m=0.10,
        max_rotation_from_anchor_rad=math.radians(10.0),
        gripper_min_width_m=0.0,
        gripper_max_width_m=0.05,
    )


@dataclass
class PiperIsaacTeleopConfig:
    """Configuration for Piper control through an Isaac node subscriber."""

    robot: PiperRobotConfig = field(default_factory=_safe_piper_robot_config)
    teleop: IsaacTeleopNodeConfig = field(default_factory=_safe_isaac_node_config)
    processor: PiperTeleopProcessorConfig = field(default_factory=_safe_piper_processor_config)
    fps: int = 30
    enable_motion: bool = False
    max_frames: int | None = None
    startup_stability_window_s: float = 1.0
    startup_stability_timeout_s: float = 10.0
    startup_max_translation_drift_m: float = 0.001
    startup_max_rotation_drift_rad: float = math.radians(0.5)

    def __post_init__(self) -> None:
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int) or self.max_frames < 0
        ):
            raise ValueError("max_frames must be None or a non-negative integer")
        if not isinstance(self.enable_motion, bool):
            raise ValueError("enable_motion must be a boolean")
        (
            self.startup_stability_window_s,
            self.startup_stability_timeout_s,
            self.startup_max_translation_drift_m,
            self.startup_max_rotation_drift_rad,
            _,
        ) = _validate_stability_options(
            stability_window_s=self.startup_stability_window_s,
            timeout_s=self.startup_stability_timeout_s,
            max_translation_drift_m=self.startup_max_translation_drift_m,
            max_rotation_drift_rad=self.startup_max_rotation_drift_rad,
            sample_period_s=None,
        )


@dataclass(frozen=True)
class TeleopStatus:
    """One renderable, testable snapshot of the teleoperation loop."""

    state: str
    hand: str
    tracking: bool
    engaged: bool
    hz: float
    measured_tcp: dict[str, float]
    target_tcp: dict[str, float]
    mode: str
    fault: str | None


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


def _validate_stability_options(
    *,
    stability_window_s: Any,
    timeout_s: Any,
    max_translation_drift_m: Any,
    max_rotation_drift_rad: Any,
    sample_period_s: Any | None,
) -> tuple[float, float, float, float, float | None]:
    window = _positive_finite(stability_window_s, "startup_stability_window_s")
    timeout = _positive_finite(timeout_s, "startup_stability_timeout_s")
    translation = _positive_finite(max_translation_drift_m, "startup_max_translation_drift_m")
    rotation = _positive_finite(max_rotation_drift_rad, "startup_max_rotation_drift_rad")
    period = None if sample_period_s is None else _positive_finite(sample_period_s, "startup sample_period_s")
    if window >= timeout:
        raise ValueError("startup_stability_window_s must be less than startup_stability_timeout_s")
    return window, timeout, translation, rotation, period


def _validate_loop_options(fps: int, max_frames: int | None) -> None:
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0
    ):
        raise ValueError("max_frames must be None or a non-negative integer")


def _complete_finite_tcp(observation: Any) -> dict[str, float] | None:
    if not isinstance(observation, Mapping):
        return None
    values: dict[str, float] = {}
    for key in _EEF_KEYS:
        value = observation.get(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        values[key] = numeric
    return values


def _tcp_drift(reference: Mapping[str, float], current: Mapping[str, float]) -> tuple[float, float]:
    reference_position = [reference[key] for key in _EEF_KEYS[:3]]
    current_position = [current[key] for key in _EEF_KEYS[:3]]
    translation = math.dist(reference_position, current_position)
    reference_rotation = Rotation.from_euler("xyz", [reference[key] for key in _EEF_KEYS[3:]])
    current_rotation = Rotation.from_euler("xyz", [current[key] for key in _EEF_KEYS[3:]])
    rotation = float((current_rotation * reference_rotation.inv()).magnitude())
    return translation, rotation


def wait_for_stable_tcp(
    robot: PiperRobot,
    *,
    stability_window_s: float,
    timeout_s: float,
    max_translation_drift_m: float,
    max_rotation_drift_rad: float,
    sample_period_s: float,
    sleep_fn: Callable[[float], None] = precise_sleep,
    clock: Callable[[], float] = time.monotonic,
) -> RobotObservation:
    """Wait for a quiet post-enable TCP window without sending any action."""

    (
        stability_window_s,
        timeout_s,
        max_translation_drift_m,
        max_rotation_drift_rad,
        validated_sample_period_s,
    ) = _validate_stability_options(
        stability_window_s=stability_window_s,
        timeout_s=timeout_s,
        max_translation_drift_m=max_translation_drift_m,
        max_rotation_drift_rad=max_rotation_drift_rad,
        sample_period_s=sample_period_s,
    )
    assert validated_sample_period_s is not None
    sample_period_s = validated_sample_period_s
    started_at = clock()
    deadline = started_at + timeout_s
    reference: dict[str, float] | None = None
    window_started_at = started_at
    while True:
        observation = robot.get_observation()
        tcp = _complete_finite_tcp(observation)
        if tcp is None:
            raise RuntimeError("Piper TCP feedback is incomplete during startup stability check.")

        now = clock()
        if now >= deadline:
            raise TimeoutError(
                f"Piper TCP did not stabilize within {timeout_s:g} seconds; no control action was sent."
            )
        if reference is None:
            reference = tcp
            window_started_at = now
        else:
            translation_drift, rotation_drift = _tcp_drift(reference, tcp)
            if translation_drift > max_translation_drift_m or rotation_drift > max_rotation_drift_rad:
                logger.info(
                    "Piper startup stability window restarted after %.3f mm / %.3f deg drift.",
                    translation_drift * 1_000.0,
                    math.degrees(rotation_drift),
                )
                reference = tcp
                window_started_at = now

        if now - window_started_at >= stability_window_s:
            return observation
        sleep_fn(min(sample_period_s, deadline - now))


def _processor_step(processor: Any) -> Any:
    steps = getattr(processor, "steps", ())
    return steps[0] if steps else None


def _status_from_frame(
    *,
    processor: Any,
    action: Mapping[str, Any],
    observation: Mapping[str, Any],
    processed_action: Mapping[str, Any],
    hz: float,
    enable_motion: bool,
) -> TeleopStatus:
    step = _processor_step(processor)
    config = getattr(step, "config", None)
    hand = getattr(config, "hand", "right")
    tracking = bool(action.get(f"{hand}.is_tracking", False))
    engaged = bool(action.get(f"{hand}.is_engaged", False))
    state = getattr(getattr(step, "state", None), "value", None) or str(getattr(step, "state", "unknown"))
    target = {key: float(processed_action[key]) for key in _EEF_KEYS if key in processed_action}
    if len(target) != len(_EEF_KEYS):
        last_target = getattr(step, "last_target", None)
        if isinstance(last_target, Mapping):
            target = {key: float(last_target[key]) for key in _EEF_KEYS if key in last_target}
    measured_tcp = _complete_finite_tcp(observation)
    fault = getattr(step, "fault_reason", None)
    if fault is None and measured_tcp is None:
        fault = "measured TCP feedback is incomplete or non-finite"
    return TeleopStatus(
        state=state,
        hand=hand,
        tracking=tracking,
        engaged=engaged,
        hz=hz,
        measured_tcp=measured_tcp or {},
        target_tcp=target,
        mode="motion" if enable_motion else "dry-run",
        fault=fault,
    )


def _format_pose(pose: Mapping[str, float]) -> str:
    if len(pose) != len(_EEF_KEYS):
        return "—"
    return " ".join(f"{pose[key]:+.3f}" for key in _EEF_KEYS)


def _render_status(status: TeleopStatus) -> Table:
    table = Table(title="Piper Isaac Teleop", expand=False)
    for heading in (
        "state",
        "hand",
        "tracking",
        "engaged",
        "Hz",
        "measured TCP",
        "target TCP",
        "mode",
        "fault",
    ):
        table.add_column(heading)
    table.add_row(
        status.state,
        status.hand,
        "yes" if status.tracking else "no",
        "yes" if status.engaged else "no",
        f"{status.hz:.1f}",
        _format_pose(status.measured_tcp),
        _format_pose(status.target_tcp),
        status.mode,
        status.fault or "—",
    )
    return table


def _attempt_termination_hold(
    robot: PiperRobot,
    observation: RobotObservation | None,
    primary_exception: BaseException | None,
) -> None:
    hold = _complete_finite_tcp(observation)
    if hold is None:
        logger.warning("Skipping termination hold because measured TCP feedback is incomplete.")
        return
    try:
        robot.send_action(hold)
    except BaseException as secondary_exception:  # nosec B110 - a safety hold is best effort
        if primary_exception is None:
            raise
        logger.warning(
            "Termination hold failed while preserving the primary exception: %s", secondary_exception
        )


def _stop_live(live: Live, primary_exception: BaseException | None) -> None:
    try:
        live.stop()
    except BaseException as secondary_exception:  # nosec B110 - preserve the loop/start exception
        if primary_exception is None:
            raise
        logger.warning(
            "Rich Live cleanup failed while preserving the primary exception: %s", secondary_exception
        )


def run_teleop_loop(
    robot: PiperRobot,
    teleop: IsaacTeleopNodeSubscriber,
    processor: RobotProcessorPipeline,
    *,
    fps: int,
    enable_motion: bool,
    max_frames: int | None = None,
    sleep_fn: Callable[[float], None] = precise_sleep,
    clock: Callable[[], float] = time.perf_counter,
    console: Console | None = None,
    status_callback: Callable[[TeleopStatus], None] | None = None,
    render: bool = True,
) -> None:
    """Run the latest-frame control loop and make one termination hold attempt."""

    _validate_loop_options(fps, max_frames)
    if not isinstance(enable_motion, bool):
        raise ValueError("enable_motion must be a boolean")

    period_s = 1.0 / fps
    frame_index = 0
    last_observation: RobotObservation | None = None
    live: Live | None = None
    live_started = False
    try:
        if render:
            live = Live(console=console or Console(), refresh_per_second=max(1, min(fps, 20)), transient=True)
            live.start()
            live_started = True
        try:
            while max_frames is None or frame_index < max_frames:
                started_at = clock()
                try:
                    observation = robot.get_observation()
                except PiperCameraTimeoutError as exc:
                    last_observation = exc.observation
                    raise
                last_observation = observation
                isaac_action = teleop.get_action()
                piper_action = processor((isaac_action, observation))
                if enable_motion and piper_action:
                    robot.send_action(piper_action)

                processing_elapsed_s = max(clock() - started_at, 0.0)
                status = _status_from_frame(
                    processor=processor,
                    action=isaac_action,
                    observation=observation,
                    processed_action=piper_action,
                    hz=1.0 / processing_elapsed_s if processing_elapsed_s > 0.0 else 0.0,
                    enable_motion=enable_motion,
                )
                if status_callback is not None:
                    status_callback(status)
                if live is not None:
                    live.update(_render_status(status))

                frame_index += 1
                if max_frames is None or frame_index < max_frames:
                    total_elapsed_s = max(clock() - started_at, 0.0)
                    sleep_fn(max(period_s - total_elapsed_s, 0.0))
        finally:
            primary_exception = sys.exception()
            if enable_motion:
                _attempt_termination_hold(robot, last_observation, primary_exception)
    finally:
        primary_exception = sys.exception()
        if live is not None and live_started:
            _stop_live(live, primary_exception)


def _disconnect_independently(
    teleop: Any,
    robot: Any,
    primary_exception: BaseException | None = None,
) -> None:
    cleanup_exception: BaseException | None = None
    for name, disconnect in (("teleop", teleop.disconnect), ("robot", robot.disconnect)):
        try:
            disconnect()
        except BaseException as error:
            if primary_exception is not None or cleanup_exception is not None:
                logger.warning("%s cleanup failed while preserving an earlier failure", name, exc_info=True)
            else:
                cleanup_exception = error
    if cleanup_exception is not None:
        raise cleanup_exception


@parser.wrap()
def teleoperate(cfg: PiperIsaacTeleopConfig) -> None:
    """Subscribe to the Isaac node, connect Piper, and run the control loop."""

    robot = PiperRobot(cfg.robot)
    teleop = IsaacTeleopNodeSubscriber(cfg.teleop)
    effective_processor_config = replace(
        cfg.processor,
        include_gripper=cfg.processor.include_gripper and cfg.robot.include_gripper,
    )
    processor = make_piper_isaac_processor(effective_processor_config)
    if not cfg.enable_motion:
        robot.config.auto_enable = False
    try:
        teleop.connect()
        robot.connect()
        if cfg.enable_motion:
            wait_for_stable_tcp(
                robot,
                stability_window_s=cfg.startup_stability_window_s,
                timeout_s=cfg.startup_stability_timeout_s,
                max_translation_drift_m=cfg.startup_max_translation_drift_m,
                max_rotation_drift_rad=cfg.startup_max_rotation_drift_rad,
                sample_period_s=1.0 / cfg.fps,
            )
            processor.reset()
        run_teleop_loop(
            robot,
            teleop,
            processor,
            fps=cfg.fps,
            enable_motion=cfg.enable_motion,
            max_frames=cfg.max_frames,
        )
    finally:
        _disconnect_independently(teleop, robot, sys.exception())


if __name__ == "__main__":
    teleoperate()
