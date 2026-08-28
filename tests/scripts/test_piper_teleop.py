from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
from draccus.argparsing import ArgumentParser
from rich.cells import cell_len
from rich.console import Console

from lekit.robots.piper import PiperCameraTimeoutError
from lekit.robots.piper.piper_robot import PiperRobotConfig
from lekit.robots.piper.teleop_processor import (
    PiperTeleopProcessorConfig,
    make_piper_isaac_processor,
)
from lekit.scripts import teleop as teleop_module
from lekit.scripts.teleop import PiperIsaacTeleopConfig, run_teleop_loop
from lekit.teleoperators.isaac_teleop import IsaacTeleopNodeConfig
from lerobot.teleoperators.config import TeleoperatorConfig

EE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")


def test_default_teleop_config_uses_validated_piper_safe_profile() -> None:
    cfg = PiperIsaacTeleopConfig()

    assert cfg.enable_motion is False
    assert cfg.robot.include_gripper is True
    assert cfg.robot.speed_percent == 10
    assert cfg.robot.max_eef_target_lead_m == pytest.approx(0.005)
    assert cfg.robot.max_eef_target_lead_rad == pytest.approx(np.deg2rad(1.0))
    assert cfg.robot.gripper_force_n == pytest.approx(1.0)
    assert cfg.robot.gripper_min_width_m == pytest.approx(0.0)
    assert cfg.robot.gripper_max_width_m == pytest.approx(0.05)
    assert isinstance(cfg.teleop, IsaacTeleopNodeConfig)
    assert cfg.teleop.endpoint == "tcp://127.0.0.1:5557"
    assert cfg.teleop.first_frame_timeout_s == pytest.approx(5.0)
    assert cfg.teleop.stale_after_s == pytest.approx(0.25)
    assert cfg.teleop.rearm_squeeze_threshold == pytest.approx(0.3)
    assert cfg.processor.include_gripper is True
    assert cfg.processor.translation_scale == pytest.approx(1.0)
    assert cfg.processor.rotation_scale == pytest.approx(1.0)
    assert cfg.processor.max_translation_from_anchor_m == pytest.approx(0.10)
    assert cfg.processor.max_rotation_from_anchor_rad == pytest.approx(np.deg2rad(10.0))
    assert cfg.processor.gripper_min_width_m == pytest.approx(0.0)
    assert cfg.processor.gripper_max_width_m == pytest.approx(0.05)


def test_safe_teleop_defaults_do_not_change_standalone_robot_defaults() -> None:
    standalone = PiperRobotConfig()
    teleop = PiperIsaacTeleopConfig()

    assert standalone.speed_percent == 20
    assert standalone.gripper_max_width_m == pytest.approx(0.07)
    assert standalone.max_eef_target_lead_rad == pytest.approx(np.deg2rad(2.0))
    assert teleop.robot is not standalone


def test_cli_decodes_piper_and_processor_string_choices() -> None:
    cfg = draccus.parse(
        PiperIsaacTeleopConfig,
        args=[
            "--robot.interface",
            "socketcan",
            "--robot.robot_model",
            "piper",
            "--robot.firmware_version",
            "v188",
            "--processor.hand",
            "left",
        ],
    )

    assert cfg.robot.interface == "socketcan"
    assert cfg.robot.robot_model == "piper"
    assert cfg.robot.firmware_version == "v188"
    assert cfg.processor.hand == "left"


def test_cli_exposes_only_independent_node_subscription_options() -> None:
    help_text = ArgumentParser(PiperIsaacTeleopConfig).parser.format_help()

    assert "--teleop.endpoint" in help_text
    assert "--teleop.first_frame_timeout_s" in help_text
    assert "--teleop.auto_launch_cloudxr" not in help_text
    assert "--teleop.cloudxr_install_dir" not in help_text
    assert "--teleop.use_head_yaw" not in help_text


def test_cli_accepts_only_the_independent_node_type_choice() -> None:
    cfg = draccus.parse(
        PiperIsaacTeleopConfig,
        args=[
            "--teleop.type",
            "isaac_teleop_node",
            "--teleop.endpoint",
            "tcp://127.0.0.1:6000",
        ],
    )

    assert isinstance(cfg.teleop, IsaacTeleopNodeConfig)
    assert cfg.teleop.endpoint == "tcp://127.0.0.1:6000"
    assert cfg.teleop.type == "isaac_teleop_node"
    assert TeleoperatorConfig.get_choice_name(IsaacTeleopNodeConfig) == "isaac_teleop_node"

    omitted_type = draccus.parse(
        PiperIsaacTeleopConfig,
        args=["--teleop.endpoint", "tcp://127.0.0.1:6001"],
    )
    assert isinstance(omitted_type.teleop, IsaacTeleopNodeConfig)
    assert omitted_type.teleop.endpoint == "tcp://127.0.0.1:6001"

    with pytest.raises(SystemExit):
        draccus.parse(
            PiperIsaacTeleopConfig,
            args=["--teleop.type", "isaac_teleop"],
        )


def measured_tcp(**overrides: float) -> dict[str, float]:
    values = dict(zip(EE_KEYS, (0.30, -0.10, 0.40, 0.10, -0.20, 0.30), strict=True))
    values.update(overrides)
    return values


class FakeRobot:
    def __init__(self, observations: list[dict[str, float]], events: list[str]) -> None:
        self.observations = iter(observations)
        self.events = events
        self.sent_actions: list[dict[str, object]] = []

    def get_observation(self) -> dict[str, float]:
        self.events.append("get_observation")
        return next(self.observations)

    def send_action(self, action: dict[str, object]) -> None:
        self.events.append("send_action")
        self.sent_actions.append(action)


class FakeTeleop:
    def __init__(self, actions: list[dict[str, object]], events: list[str]) -> None:
        self.actions = iter(actions)
        self.events = events

    def get_action(self) -> dict[str, object]:
        self.events.append("get_action")
        return next(self.actions)


class RecordingProcessor:
    def __init__(self, result: dict[str, object], events: list[str]) -> None:
        self.result = result
        self.events = events

    def __call__(self, transition: tuple[dict[str, object], dict[str, float]]) -> dict[str, object]:
        self.events.append("processor")
        return self.result


def xr_frame(*, engaged: bool, tracked: bool = True, trigger: float = 0.0) -> dict[str, object]:
    return {
        "right.translation": np.zeros(3, dtype=np.float32),
        "right.rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "right.trigger": trigger,
        "right.is_tracking": tracked,
        "right.is_engaged": engaged,
    }


def run_with_fakes(
    robot: FakeRobot,
    teleop: FakeTeleop,
    processor: Callable,
    *,
    enable_motion: bool,
    max_frames: int | None = 1,
    sleep_fn: Callable[[float], None] = lambda _duration: None,
    clock: Callable[[], float] = lambda: 0.0,
) -> None:
    run_teleop_loop(
        robot,
        teleop,
        processor,
        fps=30,
        enable_motion=enable_motion,
        max_frames=max_frames,
        sleep_fn=sleep_fn,
        clock=clock,
        render=False,
    )


def test_loop_orders_observation_before_teleop_and_processor() -> None:
    events: list[str] = []
    robot = FakeRobot([measured_tcp()], events)
    teleop = FakeTeleop([{"right.translation": "raw"}], events)
    processor = RecordingProcessor({"ee.x": 0.31}, events)

    run_with_fakes(robot, teleop, processor, enable_motion=True)

    assert events == [
        "get_observation",
        "get_action",
        "processor",
        "send_action",
        "send_action",
    ]


def test_dry_run_never_calls_send_action() -> None:
    events: list[str] = []
    robot = FakeRobot([measured_tcp(), measured_tcp()], events)
    teleop = FakeTeleop([{}, {}], events)
    processor = RecordingProcessor({"ee.x": 0.31}, events)

    run_with_fakes(robot, teleop, processor, enable_motion=False, max_frames=2)

    assert robot.sent_actions == []


def test_motion_mode_sends_only_processor_output() -> None:
    events: list[str] = []
    robot = FakeRobot([measured_tcp()], events)
    raw_action = {"right.translation": "raw", "right.is_engaged": True}
    teleop = FakeTeleop([raw_action], events)
    processed_action = {"ee.x": 0.31, "gripper.pos": 0.02}
    processor = RecordingProcessor(processed_action, events)

    run_with_fakes(robot, teleop, processor, enable_motion=True)

    assert robot.sent_actions == [processed_action, measured_tcp()]
    assert all(not key.startswith("right.") for key in robot.sent_actions[0])


def test_release_hold_output_is_sent_once() -> None:
    events: list[str] = []
    observation = measured_tcp()
    robot = FakeRobot([observation] * 4, events)
    teleop = FakeTeleop(
        [xr_frame(engaged=False), xr_frame(engaged=True), xr_frame(engaged=False), xr_frame(engaged=False)],
        events,
    )
    processor = make_piper_isaac_processor()

    run_with_fakes(robot, teleop, processor, enable_motion=True, max_frames=4)

    assert robot.sent_actions[0]["ee.x"] == pytest.approx(observation["ee.x"])
    assert robot.sent_actions[1] == observation
    assert robot.sent_actions[2] == observation
    assert len(robot.sent_actions) == 3


def test_exception_attempts_measured_tcp_hold_and_disconnects() -> None:
    events: list[str] = []
    observation = measured_tcp(**{"ee.x": 0.42})
    robot = FakeRobot([observation], events)

    class RaisingTeleop(FakeTeleop):
        def get_action(self) -> dict[str, object]:
            self.events.append("get_action")
            raise RuntimeError("teleop failed")

    teleop = RaisingTeleop([], events)
    processor = RecordingProcessor({}, events)

    with pytest.raises(RuntimeError, match="teleop failed"):
        run_with_fakes(robot, teleop, processor, enable_motion=True)

    assert robot.sent_actions == [observation]


def test_first_frame_camera_timeout_holds_exception_observation_and_reraises_same_error() -> None:
    events: list[str] = []
    fresh_observation = measured_tcp(**{"ee.x": 0.41})
    error = PiperCameraTimeoutError("wrist", fresh_observation, TimeoutError("stale frame"))

    class CameraTimeoutRobot(FakeRobot):
        def get_observation(self) -> dict[str, float]:
            self.events.append("get_observation")
            raise error

    robot = CameraTimeoutRobot([], events)

    with pytest.raises(PiperCameraTimeoutError) as exc_info:
        run_with_fakes(
            robot,
            FakeTeleop([], events),
            RecordingProcessor({}, events),
            enable_motion=True,
        )

    assert exc_info.value is error
    assert robot.sent_actions == [fresh_observation]


def test_camera_timeout_replaces_previous_frame_with_newer_exception_observation() -> None:
    events: list[str] = []
    previous_observation = measured_tcp(**{"ee.x": 0.31})
    newer_observation = measured_tcp(**{"ee.x": 0.49})
    error = PiperCameraTimeoutError("wrist", newer_observation, TimeoutError("stale frame"))

    class SecondFrameCameraTimeoutRobot(FakeRobot):
        def __init__(self) -> None:
            super().__init__([previous_observation], events)
            self.read_count = 0

        def get_observation(self) -> dict[str, float]:
            self.events.append("get_observation")
            self.read_count += 1
            if self.read_count == 2:
                raise error
            return next(self.observations)

    robot = SecondFrameCameraTimeoutRobot()

    with pytest.raises(PiperCameraTimeoutError) as exc_info:
        run_with_fakes(
            robot,
            FakeTeleop([{}], events),
            RecordingProcessor({}, events),
            enable_motion=True,
            max_frames=2,
        )

    assert exc_info.value is error
    assert robot.sent_actions == [newer_observation]
    assert robot.sent_actions[-1] != previous_observation


def test_invalid_latest_observation_skips_termination_hold_and_records_warning(caplog) -> None:
    events: list[str] = []
    invalid_observation = {**measured_tcp(), "ee.yaw": float("nan")}
    robot = FakeRobot([invalid_observation], events)
    teleop = FakeTeleop([{}], events)
    processor = RecordingProcessor({}, events)
    statuses = []

    with caplog.at_level("WARNING", logger=teleop_module.__name__):
        run_teleop_loop(
            robot,
            teleop,
            processor,
            fps=30,
            enable_motion=True,
            max_frames=1,
            render=False,
            status_callback=statuses.append,
        )

    assert robot.sent_actions == []
    assert "Skipping termination hold" in caplog.text
    assert statuses[0].fault == "measured TCP feedback is incomplete or non-finite"


def test_loop_uses_configured_frame_limit_without_sleeping_after_last_frame() -> None:
    events: list[str] = []
    robot = FakeRobot([measured_tcp(), measured_tcp()], events)
    teleop = FakeTeleop([{}, {}], events)
    processor = RecordingProcessor({}, events)
    sleeps: list[float] = []
    times = iter((0.0, 0.01, 0.02, 0.03, 0.04))

    run_with_fakes(
        robot,
        teleop,
        processor,
        enable_motion=False,
        max_frames=2,
        sleep_fn=sleeps.append,
        clock=lambda: next(times),
    )

    assert events.count("get_observation") == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": True},
        {"fps": 0},
        {"fps": 30.0},
        {"max_frames": True},
        {"max_frames": -1},
        {"max_frames": 2.0},
        {"enable_motion": 1},
    ],
)
def test_config_rejects_unsafe_loop_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PiperIsaacTeleopConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"startup_stability_window_s": -0.1},
        {"startup_stability_window_s": 0.0},
        {"startup_stability_timeout_s": 0.0},
        {"startup_stability_window_s": 2.0, "startup_stability_timeout_s": 1.0},
        {"startup_stability_window_s": 1.0, "startup_stability_timeout_s": 1.0},
        {"startup_max_translation_drift_m": 0.0},
        {"startup_max_rotation_drift_rad": float("nan")},
    ],
)
def test_config_rejects_unsafe_startup_stability_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PiperIsaacTeleopConfig(**kwargs)


def test_startup_stability_restarts_window_after_enable_settling() -> None:
    events: list[str] = []
    robot = FakeRobot(
        [
            measured_tcp(**{"ee.z": 0.4000}),
            measured_tcp(**{"ee.z": 0.3970}),
            measured_tcp(**{"ee.z": 0.3972}),
            measured_tcp(**{"ee.z": 0.3971}),
        ],
        events,
    )
    now = [0.0]

    stable_observation = teleop_module.wait_for_stable_tcp(
        robot,
        stability_window_s=0.2,
        timeout_s=1.0,
        max_translation_drift_m=0.001,
        max_rotation_drift_rad=0.01,
        sample_period_s=0.1,
        clock=lambda: now[0],
        sleep_fn=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    assert stable_observation["ee.z"] == pytest.approx(0.3971)
    assert events == ["get_observation"] * 4
    assert robot.sent_actions == []


def test_startup_stability_timeout_never_sends_a_robot_action() -> None:
    events: list[str] = []

    class DriftingRobot(FakeRobot):
        def __init__(self) -> None:
            super().__init__([], events)
            self.z = 0.4

        def get_observation(self) -> dict[str, float]:
            self.events.append("get_observation")
            self.z -= 0.002
            return measured_tcp(**{"ee.z": self.z})

    robot = DriftingRobot()
    now = [0.0]

    with pytest.raises(TimeoutError, match="did not stabilize"):
        teleop_module.wait_for_stable_tcp(
            robot,
            stability_window_s=0.2,
            timeout_s=0.3,
            max_translation_drift_m=0.001,
            max_rotation_drift_rad=0.01,
            sample_period_s=0.1,
            clock=lambda: now[0],
            sleep_fn=lambda duration: now.__setitem__(0, now[0] + duration),
        )

    assert robot.sent_actions == []


def test_startup_stability_rejects_a_stable_sample_received_after_deadline() -> None:
    events: list[str] = []
    robot = FakeRobot([measured_tcp(), measured_tcp()], events)
    times = iter((0.0, 0.0, 10.6))

    with pytest.raises(TimeoutError, match="did not stabilize"):
        teleop_module.wait_for_stable_tcp(
            robot,
            stability_window_s=1.0,
            timeout_s=10.0,
            max_translation_drift_m=0.001,
            max_rotation_drift_rad=0.01,
            sample_period_s=0.1,
            clock=lambda: next(times),
            sleep_fn=lambda _duration: None,
        )

    assert events == ["get_observation"] * 2
    assert robot.sent_actions == []


def test_startup_stability_accepts_drift_exactly_at_threshold() -> None:
    events: list[str] = []
    robot = FakeRobot(
        [measured_tcp(**{"ee.x": 0.0}), measured_tcp(**{"ee.x": 0.001})],
        events,
    )
    now = [0.0]

    observation = teleop_module.wait_for_stable_tcp(
        robot,
        stability_window_s=0.1,
        timeout_s=1.0,
        max_translation_drift_m=0.001,
        max_rotation_drift_rad=0.01,
        sample_period_s=0.1,
        clock=lambda: now[0],
        sleep_fn=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    assert observation["ee.x"] == pytest.approx(0.001)
    assert events == ["get_observation"] * 2


def test_startup_stability_uses_wrap_safe_orientation_drift() -> None:
    events: list[str] = []
    robot = FakeRobot(
        [
            measured_tcp(**{"ee.yaw": np.pi - 0.001}),
            measured_tcp(**{"ee.yaw": -np.pi + 0.001}),
        ],
        events,
    )
    now = [0.0]

    observation = teleop_module.wait_for_stable_tcp(
        robot,
        stability_window_s=0.1,
        timeout_s=1.0,
        max_translation_drift_m=0.001,
        max_rotation_drift_rad=0.003,
        sample_period_s=0.1,
        clock=lambda: now[0],
        sleep_fn=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    assert observation["ee.yaw"] == pytest.approx(-np.pi + 0.001)
    assert events == ["get_observation"] * 2


def test_startup_stability_restarts_window_after_orientation_settling() -> None:
    events: list[str] = []
    robot = FakeRobot(
        [
            measured_tcp(**{"ee.roll": 0.0000}),
            measured_tcp(**{"ee.roll": 0.0200}),
            measured_tcp(**{"ee.roll": 0.0201}),
            measured_tcp(**{"ee.roll": 0.0202}),
        ],
        events,
    )
    now = [0.0]

    observation = teleop_module.wait_for_stable_tcp(
        robot,
        stability_window_s=0.2,
        timeout_s=1.0,
        max_translation_drift_m=0.001,
        max_rotation_drift_rad=0.01,
        sample_period_s=0.1,
        clock=lambda: now[0],
        sleep_fn=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    assert observation["ee.roll"] == pytest.approx(0.0202)
    assert events == ["get_observation"] * 4


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_startup_stability_fails_closed_on_invalid_tcp_feedback(bad_value) -> None:
    events: list[str] = []
    observation = measured_tcp()
    if bad_value is None:
        observation.pop("ee.z")
    else:
        observation["ee.z"] = bad_value
    robot = FakeRobot([observation], events)

    with pytest.raises(RuntimeError, match="incomplete"):
        teleop_module.wait_for_stable_tcp(
            robot,
            stability_window_s=0.1,
            timeout_s=1.0,
            max_translation_drift_m=0.001,
            max_rotation_drift_rad=0.01,
            sample_period_s=0.1,
        )

    assert robot.sent_actions == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stability_window_s": 0.0},
        {"timeout_s": 0.0},
        {"stability_window_s": 1.0, "timeout_s": 1.0},
        {"max_translation_drift_m": 0.0},
        {"max_rotation_drift_rad": float("nan")},
        {"sample_period_s": 0.0},
    ],
)
def test_startup_stability_rejects_unsafe_direct_options(kwargs) -> None:
    events: list[str] = []
    robot = FakeRobot([measured_tcp()], events)
    options = {
        "stability_window_s": 0.1,
        "timeout_s": 1.0,
        "max_translation_drift_m": 0.001,
        "max_rotation_drift_rad": 0.01,
        "sample_period_s": 0.1,
    }
    options.update(kwargs)

    with pytest.raises(ValueError):
        teleop_module.wait_for_stable_tcp(robot, **options)

    assert events == []


class LifecycleFakeRobot:
    def __init__(self, config, events: list[str]) -> None:
        self.config = config
        self.events = events

    def connect(self) -> None:
        self.events.append("robot.connect")

    def disconnect(self) -> None:
        self.events.append("robot.disconnect")


def test_teleoperate_uses_node_subscriber_without_constructing_direct_xr(monkeypatch) -> None:
    events: list[str] = []
    robot_config = SimpleNamespace(auto_enable=True)
    robot = LifecycleFakeRobot(robot_config, events)
    teleop = LifecycleFakeTeleop(SimpleNamespace(), events)
    captured_configs: list[IsaacTeleopNodeConfig] = []

    def make_subscriber(config: IsaacTeleopNodeConfig):
        captured_configs.append(config)
        return teleop

    def reject_direct_xr(_config):
        raise AssertionError("Piper teleoperation must not construct a direct XR controller")

    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", make_subscriber, raising=False)
    monkeypatch.setattr(teleop_module, "IsaacXRController", reject_direct_xr, raising=False)
    monkeypatch.setattr(teleop_module, "make_piper_isaac_processor", lambda config: object())
    monkeypatch.setattr(teleop_module, "run_teleop_loop", lambda *args, **kwargs: events.append("loop"))

    cfg = PiperIsaacTeleopConfig(enable_motion=False)
    teleop_module.teleoperate(cfg)

    assert captured_configs == [cfg.teleop]
    assert events == ["teleop.connect", "robot.connect", "loop", "teleop.disconnect", "robot.disconnect"]


def test_teleoperate_motion_waits_for_stable_tcp_before_control(monkeypatch) -> None:
    events: list[str] = []
    captured_stability_options: dict[str, object] = {}
    robot = LifecycleFakeRobot(SimpleNamespace(auto_enable=True), events)
    teleop = LifecycleFakeTeleop(SimpleNamespace(), events)

    def record_stability_options(*args, **kwargs):
        events.append("wait_for_stable_tcp")
        captured_stability_options.update(kwargs)

    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(
        teleop_module,
        "make_piper_isaac_processor",
        lambda config: SimpleNamespace(reset=lambda: events.append("processor.reset")),
    )
    monkeypatch.setattr(
        teleop_module,
        "wait_for_stable_tcp",
        record_stability_options,
    )
    monkeypatch.setattr(teleop_module, "run_teleop_loop", lambda *args, **kwargs: events.append("loop"))

    teleop_module.teleoperate(PiperIsaacTeleopConfig(enable_motion=True))

    assert events == [
        "teleop.connect",
        "robot.connect",
        "wait_for_stable_tcp",
        "processor.reset",
        "loop",
        "teleop.disconnect",
        "robot.disconnect",
    ]
    assert captured_stability_options == {
        "stability_window_s": 1.0,
        "timeout_s": 10.0,
        "max_translation_drift_m": 0.001,
        "max_rotation_drift_rad": pytest.approx(np.deg2rad(0.5)),
        "sample_period_s": pytest.approx(1.0 / 30.0),
    }


def test_teleoperate_stability_timeout_never_enters_control_loop(monkeypatch) -> None:
    events: list[str] = []
    robot = LifecycleFakeRobot(SimpleNamespace(auto_enable=True), events)
    teleop = LifecycleFakeTeleop(SimpleNamespace(), events)

    def reject_unstable_tcp(*args, **kwargs):
        events.append("wait_for_stable_tcp")
        raise TimeoutError("Piper TCP did not stabilize")

    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(teleop_module, "make_piper_isaac_processor", lambda config: object())
    monkeypatch.setattr(teleop_module, "wait_for_stable_tcp", reject_unstable_tcp)
    monkeypatch.setattr(teleop_module, "run_teleop_loop", lambda *args, **kwargs: events.append("loop"))

    with pytest.raises(TimeoutError, match="did not stabilize"):
        teleop_module.teleoperate(PiperIsaacTeleopConfig(enable_motion=True))

    assert "loop" not in events
    assert events == [
        "teleop.connect",
        "robot.connect",
        "wait_for_stable_tcp",
        "teleop.disconnect",
        "robot.disconnect",
    ]


class LifecycleFakeTeleop:
    def __init__(self, config, events: list[str], connect_error: Exception | None = None) -> None:
        self.events = events
        self.connect_error = connect_error

    def connect(self) -> None:
        self.events.append("teleop.connect")
        if self.connect_error is not None:
            raise self.connect_error

    def disconnect(self) -> None:
        self.events.append("teleop.disconnect")


def test_teleoperate_connects_teleop_first_disables_auto_enable_and_cleans_up(monkeypatch) -> None:
    events: list[str] = []
    robot_config = SimpleNamespace(auto_enable=True)
    robot = LifecycleFakeRobot(robot_config, events)
    teleop = LifecycleFakeTeleop(SimpleNamespace(), events)

    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(teleop_module, "make_piper_isaac_processor", lambda config: object())
    monkeypatch.setattr(teleop_module, "run_teleop_loop", lambda *args, **kwargs: events.append("loop"))

    teleop_module.teleoperate(PiperIsaacTeleopConfig(enable_motion=False))

    assert robot_config.auto_enable is False
    assert events == ["teleop.connect", "robot.connect", "loop", "teleop.disconnect", "robot.disconnect"]


@pytest.mark.parametrize(
    ("robot_include_gripper", "processor_include_gripper"),
    [(False, True), (True, False)],
)
def test_teleoperate_derives_arm_only_processor_without_mutating_user_config(
    monkeypatch,
    robot_include_gripper: bool,
    processor_include_gripper: bool,
) -> None:
    events: list[str] = []
    robot_config = PiperRobotConfig(include_gripper=robot_include_gripper)
    processor_config = PiperTeleopProcessorConfig(include_gripper=processor_include_gripper)
    cfg = PiperIsaacTeleopConfig(robot=robot_config, processor=processor_config, enable_motion=False)
    robot = LifecycleFakeRobot(robot_config, events)
    teleop = LifecycleFakeTeleop(SimpleNamespace(), events)
    captured_configs: list[PiperTeleopProcessorConfig] = []
    captured_processors = []

    def make_processor(config: PiperTeleopProcessorConfig):
        captured_configs.append(config)
        processor = make_piper_isaac_processor(config)
        captured_processors.append(processor)
        return processor

    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(teleop_module, "make_piper_isaac_processor", make_processor)
    monkeypatch.setattr(teleop_module, "run_teleop_loop", lambda *args, **kwargs: events.append("loop"))

    teleop_module.teleoperate(cfg)

    effective_config = captured_configs[0]
    processor = captured_processors[0]
    released = xr_frame(engaged=False)
    engaged = xr_frame(engaged=True)
    released.pop("right.trigger")
    engaged.pop("right.trigger")
    assert processor((released, measured_tcp())) == {}
    output = processor((engaged, measured_tcp()))
    assert set(output) == set(EE_KEYS)
    assert "gripper.pos" not in output
    assert effective_config.include_gripper is False
    assert effective_config is not cfg.processor
    assert cfg.processor is processor_config
    assert cfg.processor.include_gripper is processor_include_gripper


def test_teleoperate_disconnects_robot_when_teleop_connect_fails(monkeypatch) -> None:
    events: list[str] = []
    robot = LifecycleFakeRobot(SimpleNamespace(auto_enable=True), events)
    teleop = LifecycleFakeTeleop(SimpleNamespace(), events, RuntimeError("connect failed"))

    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(teleop_module, "make_piper_isaac_processor", lambda config: object())

    with pytest.raises(RuntimeError, match="connect failed"):
        teleop_module.teleoperate(PiperIsaacTeleopConfig(enable_motion=True))

    assert events == ["teleop.connect", "teleop.disconnect", "robot.disconnect"]


def test_teleop_disconnect_failure_does_not_skip_robot_disconnect(monkeypatch) -> None:
    events: list[str] = []
    robot = LifecycleFakeRobot(SimpleNamespace(auto_enable=True), events)

    class DisconnectingTeleop(LifecycleFakeTeleop):
        def disconnect(self) -> None:
            self.events.append("teleop.disconnect")
            raise RuntimeError("teleop cleanup failed")

    teleop = DisconnectingTeleop(SimpleNamespace(), events)
    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(
        teleop_module,
        "make_piper_isaac_processor",
        lambda config: SimpleNamespace(reset=lambda: None),
    )
    monkeypatch.setattr(teleop_module, "wait_for_stable_tcp", lambda *args, **kwargs: measured_tcp())
    monkeypatch.setattr(teleop_module, "run_teleop_loop", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="teleop cleanup failed"):
        teleop_module.teleoperate(PiperIsaacTeleopConfig(enable_motion=True))

    assert events[-1] == "robot.disconnect"


@pytest.mark.parametrize("primary_stage", ["connect", "stability", "loop"])
@pytest.mark.parametrize("failing_cleanup", ["teleop", "robot"])
def test_cleanup_failure_does_not_mask_primary(
    monkeypatch,
    caplog,
    primary_stage: str,
    failing_cleanup: str,
) -> None:
    events: list[str] = []

    class CleanupRobot(LifecycleFakeRobot):
        def disconnect(self) -> None:
            super().disconnect()
            if failing_cleanup == "robot":
                raise RuntimeError("robot cleanup failed")

    class CleanupTeleop(LifecycleFakeTeleop):
        def disconnect(self) -> None:
            super().disconnect()
            if failing_cleanup == "teleop":
                raise RuntimeError("teleop cleanup failed")

    robot = CleanupRobot(SimpleNamespace(auto_enable=True), events)
    connect_error = RuntimeError("connect primary") if primary_stage == "connect" else None
    teleop = CleanupTeleop(SimpleNamespace(), events, connect_error=connect_error)
    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(teleop_module, "make_piper_isaac_processor", lambda config: object())

    def stability_gate(*args, **kwargs) -> None:
        if primary_stage == "stability":
            raise RuntimeError("stability primary")

    def control_loop(*args, **kwargs) -> None:
        if primary_stage == "loop":
            raise RuntimeError("loop primary")

    monkeypatch.setattr(teleop_module, "wait_for_stable_tcp", stability_gate)
    monkeypatch.setattr(teleop_module, "run_teleop_loop", control_loop)

    with (
        caplog.at_level("WARNING", logger=teleop_module.__name__),
        pytest.raises(RuntimeError, match=f"{primary_stage} primary"),
    ):
        teleop_module.teleoperate(PiperIsaacTeleopConfig(enable_motion=primary_stage == "stability"))

    assert events[-2:] == ["teleop.disconnect", "robot.disconnect"]
    assert f"{failing_cleanup} cleanup failed" in caplog.text


def test_loop_exception_attempts_hold_before_lifecycle_cleanup(monkeypatch) -> None:
    events: list[str] = []
    observation = measured_tcp()

    class LoopRobot(LifecycleFakeRobot):
        def get_observation(self):
            events.append("get_observation")
            return observation

        def send_action(self, action):
            events.append("send_action")

    class LoopTeleop(LifecycleFakeTeleop):
        def get_action(self):
            events.append("get_action")
            raise RuntimeError("loop failed")

    robot = LoopRobot(SimpleNamespace(auto_enable=True), events)
    teleop = LoopTeleop(SimpleNamespace(), events)
    monkeypatch.setattr(teleop_module, "PiperRobot", lambda config: robot)
    monkeypatch.setattr(teleop_module, "IsaacTeleopNodeSubscriber", lambda config: teleop)
    monkeypatch.setattr(teleop_module, "make_piper_isaac_processor", make_piper_isaac_processor)
    monkeypatch.setattr(teleop_module, "wait_for_stable_tcp", lambda *args, **kwargs: observation)

    with pytest.raises(RuntimeError, match="loop failed"):
        teleop_module.teleoperate(
            PiperIsaacTeleopConfig(
                enable_motion=True,
                max_frames=1,
            )
        )

    assert events.index("send_action") < events.index("teleop.disconnect")
    assert events.index("send_action") < events.index("robot.disconnect")


def test_status_callback_receives_control_and_tcp_snapshot() -> None:
    events: list[str] = []
    observation = measured_tcp()
    robot = FakeRobot([observation], events)
    action = {**xr_frame(engaged=True), "right.is_tracking": True}
    teleop = FakeTeleop([action], events)

    class Step:
        state = "engaged"
        fault_reason = "fault detail"
        last_target = measured_tcp(**{"ee.x": 0.31})
        config = SimpleNamespace(hand="right")

    class Processor(RecordingProcessor):
        steps = [Step()]

    target = measured_tcp(**{"ee.x": 0.31})
    processor = Processor(target, events)
    statuses = []

    run_teleop_loop(
        robot,
        teleop,
        processor,
        fps=30,
        enable_motion=False,
        max_frames=1,
        render=False,
        status_callback=statuses.append,
    )

    assert len(statuses) == 1
    status = statuses[0]
    assert status.state == "engaged"
    assert status.hand == "right"
    assert status.tracking is True
    assert status.engaged is True
    assert status.measured_tcp == observation
    assert status.target_tcp == target
    assert status.mode == "dry-run"
    assert status.fault == "fault detail"


def test_status_table_keeps_fixed_column_and_total_widths() -> None:
    expected_columns = [
        ("state", 9),
        ("hand", 5),
        ("tracking", 8),
        ("engaged", 7),
        ("Hz", 6),
        ("measured TCP", 41),
        ("target TCP", 41),
        ("mode", 7),
        ("fault", 32),
    ]

    def render(fault: str | None) -> tuple[object, list[int]]:
        table = teleop_module._render_status(
            teleop_module.TeleopStatus(
                state="engaged",
                hand="right",
                tracking=True,
                engaged=True,
                hz=29.95,
                measured_tcp=measured_tcp(),
                target_tcp=measured_tcp(**{"ee.x": 0.31}),
                mode="motion",
                fault=fault,
            )
        )
        console = Console(width=240, color_system=None)
        with console.capture() as capture:
            console.print(table)
        return table, [cell_len(line) for line in capture.get().splitlines()]

    short_table, short_widths = render(None)
    long_table, long_widths = render("controller feedback became unavailable during motion")

    for table in (short_table, long_table):
        assert table.width == 184
        assert [(column.header, column.width) for column in table.columns] == expected_columns
        assert all(column.no_wrap for column in table.columns)
        assert all(column.overflow == "ellipsis" for column in table.columns)
    assert short_widths == long_widths
    assert set(short_widths) == {184}


class FakeLive:
    def __init__(
        self, *, start_error=None, update_error=None, stop_error=None, on_update=None, **_kwargs
    ) -> None:
        self.start_error = start_error
        self.update_error = update_error
        self.stop_error = stop_error
        self.on_update = on_update
        self.started = False
        self.stop_calls = 0

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def update(self, _renderable) -> None:
        if self.on_update is not None:
            self.on_update()
        if self.update_error is not None:
            raise self.update_error

    def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        self.started = False


def test_live_stop_secondary_does_not_replace_loop_primary(monkeypatch, caplog) -> None:
    live = FakeLive(stop_error=RuntimeError("stop secondary"))
    monkeypatch.setattr(teleop_module, "Live", lambda **kwargs: live)
    robot = FakeRobot([measured_tcp()], [])

    class RaisingTeleop(FakeTeleop):
        def get_action(self):
            raise RuntimeError("loop primary")

    with (
        caplog.at_level("WARNING", logger=teleop_module.__name__),
        pytest.raises(RuntimeError, match="loop primary"),
    ):
        run_teleop_loop(
            robot,
            RaisingTeleop([{}], []),
            RecordingProcessor({}, []),
            fps=30,
            enable_motion=False,
            max_frames=1,
        )

    assert "stop secondary" in caplog.text


def test_live_start_primary_is_preserved_without_stopping_unstarted_live(monkeypatch) -> None:
    live = FakeLive(start_error=RuntimeError("start primary"), stop_error=RuntimeError("stop secondary"))
    monkeypatch.setattr(teleop_module, "Live", lambda **kwargs: live)

    with pytest.raises(RuntimeError, match="start primary"):
        run_teleop_loop(
            FakeRobot([], []),
            FakeTeleop([], []),
            RecordingProcessor({}, []),
            fps=30,
            enable_motion=False,
            max_frames=1,
        )

    assert live.stop_calls == 0


def test_live_stop_error_is_visible_without_primary(monkeypatch) -> None:
    live = FakeLive(stop_error=RuntimeError("stop primary"))
    monkeypatch.setattr(teleop_module, "Live", lambda **kwargs: live)

    with pytest.raises(RuntimeError, match="stop primary"):
        run_teleop_loop(
            FakeRobot([], []),
            FakeTeleop([], []),
            RecordingProcessor({}, []),
            fps=30,
            enable_motion=False,
            max_frames=0,
        )


def test_live_update_primary_is_preserved_when_stop_also_fails(monkeypatch, caplog) -> None:
    live = FakeLive(
        update_error=RuntimeError("update primary"),
        stop_error=RuntimeError("stop secondary"),
    )
    monkeypatch.setattr(teleop_module, "Live", lambda **kwargs: live)
    robot = FakeRobot([measured_tcp()], [])

    with (
        caplog.at_level("WARNING", logger=teleop_module.__name__),
        pytest.raises(RuntimeError, match="update primary"),
    ):
        run_teleop_loop(
            robot,
            FakeTeleop([{}], []),
            RecordingProcessor({}, []),
            fps=30,
            enable_motion=False,
            max_frames=1,
        )

    assert "stop secondary" in caplog.text


def test_pacing_includes_status_and_live_update_time(monkeypatch) -> None:
    now = [0.0]
    sleeps: list[float] = []
    live = FakeLive(on_update=lambda: now.__setitem__(0, now[0] + 0.02))
    monkeypatch.setattr(teleop_module, "Live", lambda **kwargs: live)
    robot = FakeRobot([measured_tcp(), measured_tcp()], [])

    def status_callback(_status) -> None:
        now[0] += 0.01

    run_teleop_loop(
        robot,
        FakeTeleop([{}, {}], []),
        RecordingProcessor({}, []),
        fps=30,
        enable_motion=False,
        max_frames=2,
        clock=lambda: now[0],
        sleep_fn=sleeps.append,
        status_callback=status_callback,
    )

    assert sleeps == [pytest.approx(1.0 / 30.0 - 0.03)]


class RaisingRobot(FakeRobot):
    def __init__(self, observations, events, *, send_error=None):
        super().__init__(observations, events)
        self.send_error = send_error

    def send_action(self, action):
        self.events.append("send_action")
        self.sent_actions.append(action)
        if self.send_error is not None:
            raise self.send_error


@pytest.mark.parametrize(
    "failure_point", ["get_observation", "get_action", "processor", "send", "status", "sleep"]
)
def test_loop_failures_preserve_primary_and_attempt_latest_hold(failure_point) -> None:
    events: list[str] = []
    observation = measured_tcp()

    class ObservationRobot(FakeRobot):
        def __init__(self):
            super().__init__([observation], events)
            self.reads = 0

        def get_observation(self):
            self.reads += 1
            if failure_point == "get_observation" and self.reads == 2:
                raise RuntimeError("get_observation primary")
            return observation

    class ActionTeleop(FakeTeleop):
        def get_action(self):
            if failure_point == "get_action":
                raise RuntimeError("get_action primary")
            return {}

    class Processor:
        def __call__(self, _transition):
            if failure_point == "processor":
                raise RuntimeError("processor primary")
            return {"ee.x": 0.31}

    if failure_point == "send":
        robot = RaisingRobot([observation], events, send_error=RuntimeError("send primary"))
    else:
        robot = ObservationRobot()
    teleop = ActionTeleop([], events)
    processor = Processor()
    if failure_point == "status":

        def callback(_status):
            raise RuntimeError("status primary")
    else:
        callback = None
    if failure_point == "sleep":

        def sleep_fn(_duration):
            raise RuntimeError("sleep primary")

        max_frames = 2
        robot = ObservationRobot()
    else:

        def sleep_fn(_duration):
            return None

        max_frames = 2 if failure_point == "get_observation" else 1

    expected = f"{failure_point} primary"
    with pytest.raises(RuntimeError, match=expected):
        run_teleop_loop(
            robot,
            teleop,
            processor,
            fps=30,
            enable_motion=True,
            max_frames=max_frames,
            sleep_fn=sleep_fn,
            render=False,
            status_callback=callback,
        )

    assert robot.sent_actions, "termination hold was not attempted"
    assert robot.sent_actions[-1] == observation


def test_termination_hold_failure_does_not_replace_loop_primary(caplog) -> None:
    observation = measured_tcp()
    robot = RaisingRobot([observation], [], send_error=RuntimeError("hold secondary"))

    class RaisingTeleop(FakeTeleop):
        def get_action(self):
            raise RuntimeError("loop primary")

    with (
        caplog.at_level("WARNING", logger=teleop_module.__name__),
        pytest.raises(RuntimeError, match="loop primary"),
    ):
        run_teleop_loop(
            robot,
            RaisingTeleop([], []),
            RecordingProcessor({}, []),
            fps=30,
            enable_motion=True,
            max_frames=1,
            render=False,
        )

    assert "hold secondary" in caplog.text


def test_termination_hold_keyboard_interrupt_is_visible_without_primary() -> None:
    observation = measured_tcp()
    robot = RaisingRobot([observation], [], send_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_teleop_loop(
            robot,
            FakeTeleop([{}], []),
            RecordingProcessor({}, []),
            fps=30,
            enable_motion=True,
            max_frames=1,
            render=False,
        )
