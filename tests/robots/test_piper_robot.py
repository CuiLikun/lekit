from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lekit.robots.piper import piper_robot as driver
from lekit.robots.piper.cartesian_servo import (
    PiperCartesianServoConfig,
    PiperCartesianServoDiagnostics,
    PiperCartesianServoStep,
)

CURRENT_TIME_S = 1_700_000_000.0


@pytest.fixture(autouse=True)
def current_wall_clock(monkeypatch):
    monkeypatch.setattr(driver.time, "time", lambda: CURRENT_TIME_S)


class FakeEffector:
    def __init__(self) -> None:
        self.width_m = 0.04
        self.mode = "width"
        self.moves: list[tuple[float, float]] = []

    def get_gripper_status(self):
        message = SimpleNamespace(
            value=self.width_m,
            force=1.0,
            mode=self.mode,
            foc_status=SimpleNamespace(
                voltage_too_low=False,
                motor_overheating=False,
                driver_overcurrent=False,
                driver_overheating=False,
                sensor_status=False,
                driver_error_status=False,
                driver_enable_status=True,
                homing_status=False,
            ),
        )
        return SimpleNamespace(msg=message, hz=100.0, timestamp=CURRENT_TIME_S)

    def move_gripper_m(self, *, value: float, force: float) -> None:
        self.width_m = value
        self.moves.append((value, force))


class FakeArm:
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="agx_gripper"))

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.connected = False
        self.joints = [0.0, 0.1, -0.2, 0.3, 0.4, 0.5]
        self.tcp_pose = [0.35, -0.10, 0.30, 0.10, -0.20, 0.30]
        self.tcp_feedback_timestamp: object = CURRENT_TIME_S
        self.tcp_offset = [0.0] * 6
        self.healthy = True
        self.fk_result = [0.35, -0.10, 0.20, 0.10, -0.20, 0.30]
        self.fk_error = None
        self.flange2tcp_result = None
        self.flange2tcp_error = None
        self.effector = FakeEffector()
        self.joint_limits_enabled = False
        self.auto_set_motion_mode_enabled = True
        self._parser = SimpleNamespace(
            joint_12=object(),
            joint_34=object(),
            joint_56=object(),
            end_pose_xy=SimpleNamespace(timestamp=CURRENT_TIME_S),
            end_pose_zrx=SimpleNamespace(timestamp=CURRENT_TIME_S),
            end_pose_ryrz=SimpleNamespace(timestamp=CURRENT_TIME_S),
        )

    def set_tcp_timestamps(self, timestamp: object) -> None:
        self.tcp_feedback_timestamp = timestamp
        self._parser.end_pose_xy = SimpleNamespace(timestamp=timestamp)
        self._parser.end_pose_zrx = SimpleNamespace(timestamp=timestamp)
        self._parser.end_pose_ryrz = SimpleNamespace(timestamp=timestamp)

    def init_effector(self, kind: str):
        self.events.append(("init_effector", kind))
        return self.effector

    def connect(self) -> None:
        self.events.append(("connect",))
        self.connected = True

    def disconnect(self) -> None:
        self.events.append(("disconnect",))
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def is_ok(self) -> bool:
        self.events.append(("is_ok",))
        return self.healthy

    def enable(self) -> bool:
        self.events.append(("enable",))
        return True

    def disable(self) -> bool:
        self.events.append(("disable",))
        return True

    def set_speed_percent(self, percent: int) -> None:
        self.events.append(("set_speed_percent", percent))

    def set_auto_set_motion_mode_enabled(self, enabled: bool) -> None:
        self.auto_set_motion_mode_enabled = enabled
        self.events.append(("set_auto_set_motion_mode_enabled", enabled))

    def set_motion_mode(self, mode: str) -> None:
        self.events.append(("set_motion_mode", mode))

    def set_joint_limits_enabled(self, enabled: bool) -> None:
        self.joint_limits_enabled = enabled
        self.events.append(("set_joint_limits_enabled", enabled))

    def get_joint_limits_enabled(self) -> bool:
        return self.joint_limits_enabled

    def get_config(self) -> dict:
        return {
            "joint_limits": {
                "joint1": [-2.617994, 2.617994],
                "joint2": [0.0, 3.141593],
                "joint3": [-2.967060, 0.0],
                "joint4": [-1.745330, 1.745330],
                "joint5": [-1.221730, 1.221730],
                "joint6": [-2.094396, 2.094396],
            }
        }

    def get_joint_angles(self):
        return SimpleNamespace(msg=list(self.joints), hz=200.0, timestamp=CURRENT_TIME_S)

    def set_tcp_offset(self, pose: list[float]) -> None:
        self.tcp_offset = list(pose)
        self.events.append(("set_tcp_offset", list(pose)))

    def get_tcp_pose(self):
        return SimpleNamespace(
            msg=list(self.tcp_pose),
            hz=100.0,
            timestamp=self.tcp_feedback_timestamp,
        )

    def fk(self, joints: list[float]) -> list[float]:
        self.events.append(("fk", list(joints)))
        if self.fk_error is not None:
            raise self.fk_error
        return self.fk_result

    def get_flange2tcp_pose(self, flange_pose: list[float]) -> list[float]:
        self.events.append(("get_flange2tcp_pose", list(flange_pose)))
        if self.flange2tcp_error is not None:
            raise self.flange2tcp_error
        if self.flange2tcp_result is not None:
            return self.flange2tcp_result
        return [flange_pose[0], flange_pose[1], flange_pose[2] + 0.10, *flange_pose[3:]]

    def move_p(self, flange_pose: list[float]) -> None:
        if self.auto_set_motion_mode_enabled:
            self.set_motion_mode("p")
        self.events.append(("move_p", list(flange_pose)))

    def move_j(self, joints: list[float]) -> None:
        if self.auto_set_motion_mode_enabled:
            self.set_motion_mode("j")
        self.joints = list(joints)
        self.events.append(("move_j", list(joints)))


class FakeCartesianServo:
    instances: list[FakeCartesianServo] = []

    def __init__(self, config, *, joint_limits, fk_tcp, **_kwargs) -> None:
        self.config = config
        self.joint_limits = tuple(joint_limits)
        self.fk_tcp = fk_tcp
        self.reset_count = 0
        self.steps: list[tuple[list[float], list[float], list[float], float]] = []
        self.instances.append(self)

    def reset(self) -> None:
        self.reset_count += 1

    def step(self, measured_joints, measured_tcp, target_tcp, *, dt) -> PiperCartesianServoStep:
        measured_joints = list(measured_joints)
        measured_tcp = list(measured_tcp)
        target_tcp = list(target_tcp)
        self.steps.append((measured_joints, measured_tcp, target_tcp, dt))
        self.fk_tcp(measured_joints)
        diagnostics = PiperCartesianServoDiagnostics(
            state="tracking",
            minimum_singular_value=0.02,
            damping=0.12,
            orientation_scale=0.25,
            position_error_m=0.01,
            orientation_error_rad=0.02,
            maximum_joint_velocity_rad_s=0.1,
            solver_duration_ms=0.2,
            joint_limit_saturated=False,
        )
        return PiperCartesianServoStep((0.01, 0.1, -0.2, 0.3, 0.0, 0.5), diagnostics)


@pytest.fixture(autouse=True)
def fake_cartesian_servo(monkeypatch):
    FakeCartesianServo.instances.clear()
    monkeypatch.setattr(driver, "PiperCartesianServo", FakeCartesianServo, raising=False)


class FakeCamera:
    width = 8
    height = 6
    use_rgb = True
    use_depth = False

    def __init__(self) -> None:
        self.is_connected = False

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def read_latest(self) -> np.ndarray:
        return np.full((6, 8, 3), 7, dtype=np.uint8)


class TimeoutCamera(FakeCamera):
    def read_latest(self) -> np.ndarray:
        raise TimeoutError("stale frame")


@pytest.fixture
def robot(monkeypatch):
    arm = FakeArm()
    sdk_configs: list[dict] = []

    def create_sdk_arm(config):
        sdk_configs.append(dict(config))
        return arm

    monkeypatch.setattr(driver, "create_piper_sdk_arm", create_sdk_arm)
    config = driver.PiperRobotConfig(channel="can-test", max_relative_target=0.05)
    robot = driver.PiperRobot(config)
    yield robot, arm, sdk_configs
    robot.disconnect()


def test_features_match_six_joint_gripper_and_camera_contract(monkeypatch):
    monkeypatch.setattr(driver, "make_cameras_from_configs", lambda _configs: {"wrist": FakeCamera()})
    config = driver.PiperRobotConfig(cameras={"wrist": SimpleNamespace(width=8, height=6, fps=30)})

    robot = driver.PiperRobot(config)

    expected = {f"joint_{index}.pos": float for index in range(1, 7)}
    expected["gripper.pos"] = float
    expected.update(dict.fromkeys(("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw"), float))
    assert robot.action_features == expected
    assert robot.observation_features == {**expected, "wrist": (6, 8, 3)}


def test_tcp_config_normalizes_valid_values():
    config = driver.PiperRobotConfig(
        tcp_offset=(1, "-2", 3, math.pi, math.pi / 2.0, -math.pi),
        eef_workspace_min_m=("-0.5", -0.4, 0.03),
        eef_workspace_max_m=(0.5, "0.4", 0.7),
        max_eef_target_lead_m="0.0",
        max_eef_target_lead_rad=2,
    )

    assert config.tcp_offset == (1.0, -2.0, 3.0, math.pi, math.pi / 2.0, -math.pi)
    assert config.eef_workspace_min_m == (-0.5, -0.4, 0.03)
    assert config.eef_workspace_max_m == (0.5, 0.4, 0.7)
    assert config.max_eef_target_lead_m == 0.0
    assert config.max_eef_target_lead_rad == 2.0


def test_cartesian_servo_config_normalizes_json_mapping_and_rejects_other_types():
    config = driver.PiperRobotConfig(cartesian_servo={"max_joint_velocity_rad_s": 0.25})

    assert isinstance(config.cartesian_servo, PiperCartesianServoConfig)
    assert config.cartesian_servo.max_joint_velocity_rad_s == pytest.approx(0.25)
    with pytest.raises(TypeError, match="cartesian_servo"):
        driver.PiperRobotConfig(cartesian_servo=object())


def test_tcp_feedback_max_age_defaults_to_point_one_seconds():
    assert driver.PiperRobotConfig().tcp_feedback_max_age_s == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interface", "invalid"),
        ("robot_model", "invalid"),
        ("firmware_version", "invalid"),
    ],
)
def test_sdk_string_choice_configuration_rejects_unknown_values(field, value):
    with pytest.raises(ValueError, match=field):
        driver.PiperRobotConfig(**{field: value})


@pytest.mark.parametrize("value", [True, 0.0, -0.1, float("nan"), float("inf"), "invalid"])
def test_tcp_feedback_max_age_rejects_non_positive_or_non_finite_values(value):
    with pytest.raises(ValueError, match="tcp_feedback_max_age_s"):
        driver.PiperRobotConfig(tcp_feedback_max_age_s=value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tcp_offset": (0, 1, 2, 3, 4)}, "tcp_offset"),
        ({"tcp_offset": (0, 1, 2, 3, 4, float("nan"))}, "tcp_offset"),
        ({"eef_workspace_min_m": (0, 0), "eef_workspace_max_m": (1, 1, 1)}, "workspace"),
        ({"eef_workspace_min_m": (0, 0, 1), "eef_workspace_max_m": (1, 1, 1)}, "workspace"),
        ({"eef_workspace_min_m": (0, 0, 0), "eef_workspace_max_m": (1, float("inf"), 1)}, "workspace"),
        ({"max_eef_target_lead_m": -0.01}, "max_eef_target_lead_m"),
        ({"max_eef_target_lead_rad": float("nan")}, "max_eef_target_lead_rad"),
        ({"max_eef_target_lead_rad": "invalid"}, "max_eef_target_lead_rad"),
    ],
)
def test_tcp_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        driver.PiperRobotConfig(**kwargs)


@pytest.mark.parametrize(
    "tcp_offset",
    [
        (0.0, 0.0, 0.0, -math.pi, -math.pi / 2.0, -math.pi),
        (0.0, 0.0, 0.0, math.pi, math.pi / 2.0, math.pi),
    ],
)
def test_tcp_offset_accepts_exact_sdk_euler_boundaries(tcp_offset):
    assert driver.PiperRobotConfig(tcp_offset=tcp_offset).tcp_offset == tcp_offset


@pytest.mark.parametrize(
    "tcp_offset",
    [
        (0.0, 0.0, 0.0, math.pi + 1e-12, 0.0, 0.0),
        (0.0, 0.0, 0.0, -math.pi - 1e-12, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, math.pi / 2.0 + 1e-12, 0.0),
        (0.0, 0.0, 0.0, 0.0, -math.pi / 2.0 - 1e-12, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, math.pi + 1e-12),
        (0.0, 0.0, 0.0, 0.0, 0.0, -math.pi - 1e-12),
    ],
)
def test_tcp_offset_rejects_values_just_outside_sdk_euler_boundaries(tcp_offset):
    with pytest.raises(ValueError, match="tcp_offset"):
        driver.PiperRobotConfig(tcp_offset=tcp_offset)


def test_connect_applies_configured_tcp_offset(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    config = driver.PiperRobotConfig(tcp_offset=(0.01, -0.02, 0.03, 0.04, -0.05, 0.06))
    piper = driver.PiperRobot(config)

    try:
        piper.connect()
    finally:
        piper.disconnect()

    assert arm.tcp_offset == [0.01, -0.02, 0.03, 0.04, -0.05, 0.06]
    assert ("set_tcp_offset", [0.01, -0.02, 0.03, 0.04, -0.05, 0.06]) in arm.events


def test_tcp_feedback_is_returned_in_observation(robot):
    piper, _arm, _sdk_configs = robot
    piper.connect()

    observation = piper.get_observation()

    assert observation["ee.x"] == pytest.approx(0.35)
    assert observation["ee.y"] == pytest.approx(-0.10)
    assert observation["ee.z"] == pytest.approx(0.30)
    assert observation["ee.roll"] == pytest.approx(0.10)
    assert observation["ee.pitch"] == pytest.approx(-0.20)
    assert observation["ee.yaw"] == pytest.approx(0.30)


@pytest.mark.parametrize(
    "tcp_pose",
    [
        [0.35, -0.10, 0.30],
        [0.35, "invalid", 0.30, 0.10, -0.20, 0.30],
        [0.35, -0.10, float("nan"), 0.10, -0.20, 0.30],
    ],
)
def test_malformed_or_non_finite_tcp_feedback_raises_piper_feedback_error(robot, tcp_pose):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.tcp_pose = tcp_pose

    with pytest.raises(driver.PiperFeedbackError, match="TCP pose feedback"):
        piper.get_observation()


@pytest.mark.parametrize("frame_name", ["end_pose_xy", "end_pose_zrx", "end_pose_ryrz"])
def test_tcp_feedback_rejects_each_missing_sdk_component_frame(robot, frame_name):
    piper, arm, _sdk_configs = robot
    piper.connect()
    setattr(arm._parser, frame_name, None)

    with pytest.raises(driver.PiperFeedbackError, match=frame_name):
        piper.get_observation()


def test_tcp_feedback_rejects_missing_sdk_parser(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm._parser = None

    with pytest.raises(driver.PiperFeedbackError, match="parser"):
        piper.get_observation()


@pytest.mark.parametrize("offset_s", [-0.101, 0.101])
def test_tcp_feedback_rejects_stale_or_future_sdk_component_frame(robot, offset_s):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm._parser.end_pose_zrx.timestamp = CURRENT_TIME_S + offset_s

    with pytest.raises(driver.PiperFeedbackError, match="end_pose_zrx.*timestamp"):
        piper.get_observation()


@pytest.mark.parametrize("offset_s", [-0.101, 0.101])
def test_tcp_feedback_rejects_stale_or_future_synthesized_timestamp(robot, offset_s):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.tcp_feedback_timestamp = CURRENT_TIME_S + offset_s

    with pytest.raises(driver.PiperFeedbackError, match="TCP pose feedback timestamp"):
        piper.get_observation()


@pytest.mark.parametrize("offset_s", [-0.1, 0.1])
def test_tcp_feedback_accepts_timestamps_at_exact_age_boundary(robot, offset_s):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.set_tcp_timestamps(CURRENT_TIME_S + offset_s)

    observation = piper.get_observation()

    assert observation["ee.x"] == pytest.approx(arm.tcp_pose[0])


def test_tcp_feedback_wraps_sdk_call_exception(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()

    def fail_get_tcp_pose():
        raise RuntimeError("SDK read failed")

    arm.get_tcp_pose = fail_get_tcp_pose

    with pytest.raises(driver.PiperFeedbackError, match="SDK read") as exc_info:
        piper.get_observation()

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_tcp_feedback_wraps_timestamp_conversion_exception(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.tcp_feedback_timestamp = object()

    with pytest.raises(driver.PiperFeedbackError, match="TCP pose feedback timestamp") as exc_info:
        piper.get_observation()

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_connect_builds_documented_sdk_config_and_initializes_effector_before_connection(robot):
    piper, arm, sdk_configs = robot

    piper.connect()

    assert sdk_configs == [
        {
            "robot": "piper",
            "firmeware_version": "default",
            "channel": "can-test",
            "interface": "socketcan",
            "bitrate": 1_000_000,
        }
    ]
    assert arm.events == [
        ("init_effector", "agx_gripper"),
        ("connect",),
        ("set_auto_set_motion_mode_enabled", False),
        ("set_joint_limits_enabled", True),
        ("set_speed_percent", 20),
        ("set_tcp_offset", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("enable",),
    ]
    assert piper.is_connected is True


def test_connect_enables_sdk_model_joint_limits(robot):
    piper, arm, _sdk_configs = robot

    piper.connect()

    assert arm.joint_limits_enabled is True
    assert ("set_joint_limits_enabled", True) in arm.events


def test_connect_waits_for_initial_joint_and_gripper_feedback(monkeypatch):
    arm = FakeArm()
    arm._parser = SimpleNamespace(
        joint_12=None,
        joint_34=None,
        joint_56=None,
        end_pose_xy=SimpleNamespace(timestamp=CURRENT_TIME_S),
        end_pose_zrx=SimpleNamespace(timestamp=CURRENT_TIME_S),
        end_pose_ryrz=SimpleNamespace(timestamp=CURRENT_TIME_S),
    )
    joint_reads = 0
    gripper_reads = 0
    read_joints = arm.get_joint_angles
    read_gripper = arm.effector.get_gripper_status

    def delayed_joints():
        nonlocal joint_reads
        joint_reads += 1
        if joint_reads == 1:
            arm._parser.joint_12 = object()
            return None
        if joint_reads == 2:
            arm._parser.joint_34 = object()
        else:
            arm._parser.joint_56 = object()
        return read_joints()

    def delayed_gripper():
        nonlocal gripper_reads
        gripper_reads += 1
        return None if gripper_reads == 1 else read_gripper()

    arm.get_joint_angles = delayed_joints
    arm.effector.get_gripper_status = delayed_gripper
    monkeypatch.setattr(driver.time, "sleep", lambda _duration: None)
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig(enable_poll_interval_s=0.001))

    try:
        piper.connect()
    finally:
        piper.disconnect()

    assert joint_reads == 3
    assert gripper_reads == 2


def test_connect_waits_for_complete_fresh_tcp_before_configure_and_enable(monkeypatch):
    arm = FakeArm()
    arm.set_tcp_timestamps(CURRENT_TIME_S - 1.0)
    sleep_calls = 0

    def refresh_tcp_after_first_poll(_duration: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        arm.set_tcp_timestamps(CURRENT_TIME_S)

    monkeypatch.setattr(driver.time, "sleep", refresh_tcp_after_first_poll)
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig())

    try:
        piper.connect()
    finally:
        piper.disconnect()

    configure_index = next(
        index for index, event in enumerate(arm.events) if event[0] == "set_joint_limits_enabled"
    )
    enable_index = next(index for index, event in enumerate(arm.events) if event[0] == "enable")
    assert sleep_calls == 1
    assert configure_index < enable_index


def test_initial_stale_tcp_timeout_never_configures_or_enables(monkeypatch):
    arm = FakeArm()
    arm.set_tcp_timestamps(CURRENT_TIME_S - 1.0)
    monotonic_now = [0.0]

    monkeypatch.setattr(driver.time, "monotonic", lambda: monotonic_now[0])
    monkeypatch.setattr(
        driver.time,
        "sleep",
        lambda duration: monotonic_now.__setitem__(0, monotonic_now[0] + duration),
    )
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig(feedback_timeout_s=0.02, feedback_poll_interval_s=0.01))

    with pytest.raises(TimeoutError, match="initial feedback"):
        piper.connect()

    assert not any(
        event[0] in {"set_joint_limits_enabled", "set_speed_percent", "set_tcp_offset", "enable"}
        for event in arm.events
    )


def test_observation_uses_sdk_si_units_and_reads_camera(robot):
    piper, _arm, _sdk_configs = robot
    camera = FakeCamera()
    piper.cameras["wrist"] = camera
    piper.connect()

    observation = piper.get_observation()

    assert [observation[f"joint_{index}.pos"] for index in range(1, 7)] == pytest.approx(
        [0.0, 0.1, -0.2, 0.3, 0.4, 0.5]
    )
    assert observation["gripper.pos"] == pytest.approx(0.04)
    assert observation["wrist"].shape == (6, 8, 3)
    assert np.all(observation["wrist"] == 7)


def test_tcp_action_uses_cartesian_servo_and_move_j(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()

    applied = piper.send_action(
        {
            "ee.x": 0.45,
            "ee.y": -0.20,
            "ee.z": 0.40,
            "ee.roll": 0.20,
            "ee.pitch": -0.30,
            "ee.yaw": 0.40,
            "gripper.pos": 0.02,
        }
    )

    servo = FakeCartesianServo.instances[-1]
    measured_joints, measured_tcp, tcp_target, dt = servo.steps[-1]

    assert measured_joints == pytest.approx([0.0, 0.1, -0.2, 0.3, 0.4, 0.5])
    assert measured_tcp == pytest.approx(arm.tcp_pose)
    assert np.linalg.norm(np.subtract(tcp_target[:3], arm.tcp_pose[:3])) <= 0.005 + 1e-12
    assert dt == pytest.approx(1.0 / 30.0)
    move_target = next(event[1] for event in arm.events if event[0] == "move_j")
    assert not any(event[0] == "move_p" for event in arm.events)
    assert [applied[key] for key in piper._JOINT_KEYS] == pytest.approx(move_target)
    assert applied["joint_5.pos"] == pytest.approx(0.0)
    assert [applied[key] for key in piper._EEF_KEYS] == pytest.approx(tcp_target)
    assert arm.effector.moves == [(0.02, 1.0)]


def test_eef_motion_mode_selects_joint_mode_once_across_repeated_frames(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.events.clear()
    tcp_action = dict(zip(piper._EEF_KEYS, arm.tcp_pose, strict=True))

    piper.send_action(tcp_action)
    piper.send_action(tcp_action)

    assert [event for event in arm.events if event[0] == "set_motion_mode"] == [
        ("set_motion_mode", "j"),
    ]


def test_cartesian_servo_receives_sdk_fk_tcp_adapter_and_model_limits(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    servo = FakeCartesianServo.instances[-1]
    joints = [0.01, 0.1, -0.2, 0.3, 0.0, 0.5]

    tcp = servo.fk_tcp(joints)

    fk_index = next(index for index, event in enumerate(arm.events) if event == ("fk", joints))
    tcp_index = next(
        index
        for index, event in enumerate(arm.events)
        if event == ("get_flange2tcp_pose", arm.fk_result)
    )
    assert fk_index < tcp_index
    assert tcp == pytest.approx([0.35, -0.10, 0.30, 0.10, -0.20, 0.30])
    assert servo.joint_limits == (
        (-2.617994, 2.617994),
        (0.0, 3.141593),
        (-2.967060, 0.0),
        (-1.745330, 1.745330),
        (-1.221730, 1.221730),
        (-2.094396, 2.094396),
    )


def test_requested_gripper_is_prepared_before_servo_motion_and_sent_after(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.events.clear()
    servo = FakeCartesianServo.instances[-1]

    class RecordingWidth(float):
        def __gt__(self, other):
            arm.events.append(("prepare_gripper",))
            return super().__gt__(other)

        def __lt__(self, other):
            arm.events.append(("prepare_gripper",))
            return super().__lt__(other)

    piper.config.gripper_min_width_m = RecordingWidth(0.0)
    piper.config.gripper_max_width_m = RecordingWidth(0.07)
    original_step = servo.step
    original_move_gripper = arm.effector.move_gripper_m

    def recording_step(*args, **kwargs):
        arm.events.append(("servo_step",))
        return original_step(*args, **kwargs)

    def recording_move_gripper(*, value: float, force: float) -> None:
        arm.events.append(("move_gripper", value, force))
        original_move_gripper(value=value, force=force)

    servo.step = recording_step
    arm.effector.move_gripper_m = recording_move_gripper
    action = dict(zip(piper._EEF_KEYS, arm.tcp_pose, strict=True))

    applied = piper.send_action(action | {"gripper.pos": 0.20})

    prepare_index = next(index for index, event in enumerate(arm.events) if event[0] == "prepare_gripper")
    step_index = next(index for index, event in enumerate(arm.events) if event[0] == "servo_step")
    move_index = next(index for index, event in enumerate(arm.events) if event[0] == "move_j")
    gripper_index = next(index for index, event in enumerate(arm.events) if event[0] == "move_gripper")
    assert prepare_index < step_index < move_index < gripper_index
    assert arm.events[gripper_index][1:] == pytest.approx((0.07, 1.0))
    assert applied["gripper.pos"] == pytest.approx(0.07)


@pytest.mark.parametrize(
    "fk_result",
    [
        [0.0] * 5,
        ["invalid", 0.0, 0.0, 0.0, 0.0, 0.0],
        [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_malformed_fk_output_rejects_cartesian_action_before_move_j(robot, fk_result):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.events.clear()
    arm.effector.moves.clear()
    arm.fk_result = fk_result

    with pytest.raises(driver.PiperFeedbackError, match="Cartesian servo FK failed"):
        piper.send_action(dict(zip(piper._EEF_KEYS, arm.tcp_pose, strict=True)) | {"gripper.pos": 0.02})

    assert any(event[0] == "fk" for event in arm.events)
    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_switching_from_eef_to_explicit_joints_resets_cartesian_servo(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    servo = FakeCartesianServo.instances[-1]
    initial_resets = servo.reset_count

    piper.send_action(dict(zip(piper._EEF_KEYS, arm.tcp_pose, strict=True)))
    piper.send_action({"joint_1.pos": arm.joints[0]})

    assert servo.reset_count == initial_resets + 1


def test_connect_and_disconnect_reset_and_clear_cartesian_servo(robot):
    piper, arm, _sdk_configs = robot

    piper.connect()
    servo = FakeCartesianServo.instances[-1]
    assert servo.reset_count == 1
    piper.send_action(dict(zip(piper._EEF_KEYS, arm.tcp_pose, strict=True)))
    assert piper.cartesian_servo_status()

    piper.disconnect()

    assert servo.reset_count == 2
    assert piper._cartesian_servo is None
    assert piper.cartesian_servo_status() == {}


def test_cartesian_servo_status_is_plain_mapping_and_empty_before_first_step(robot):
    piper, arm, _sdk_configs = robot
    assert piper.cartesian_servo_status() == {}
    piper.connect()
    assert piper.cartesian_servo_status() == {}

    piper.send_action(dict(zip(piper._EEF_KEYS, arm.tcp_pose, strict=True)))

    status = piper.cartesian_servo_status()
    assert type(status) is dict
    assert status == {
        "state": "tracking",
        "minimum_singular_value": 0.02,
        "damping": 0.12,
        "orientation_scale": 0.25,
        "position_error_m": 0.01,
        "orientation_error_rad": 0.02,
        "maximum_joint_velocity_rad_s": 0.1,
        "solver_duration_ms": 0.2,
        "joint_limit_saturated": False,
    }


def test_joint_and_tcp_fields_cannot_be_mixed(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()

    with pytest.raises(ValueError, match="mix"):
        piper.send_action({"joint_1.pos": 0.01, **dict.fromkeys(piper._EEF_KEYS, 0.0)})

    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_partial_tcp_action_is_rejected(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()

    with pytest.raises(ValueError, match="incomplete TCP pose"):
        piper.send_action({"ee.x": 0.36})

    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_invalid_gripper_rejects_before_cartesian_motion(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()

    with pytest.raises(ValueError, match="gripper.pos"):
        piper.send_action({**dict.fromkeys(piper._EEF_KEYS, 0.0), "gripper.pos": float("nan")})

    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_tcp_translation_is_clamped_to_workspace(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(
        driver.PiperRobotConfig(
            max_eef_target_lead_m=None,
            max_eef_target_lead_rad=None,
            eef_workspace_min_m=(0.0, -0.5, 0.10),
            eef_workspace_max_m=(0.40, 0.5, 0.50),
        )
    )
    piper.connect()

    piper.send_action(
        {
            "ee.x": 9.0,
            "ee.y": 9.0,
            "ee.z": -9.0,
            "ee.roll": 0.10,
            "ee.pitch": -0.20,
            "ee.yaw": 0.30,
        }
    )

    tcp_target = FakeCartesianServo.instances[-1].steps[-1][2]
    assert tcp_target[:3] == pytest.approx([0.40, 0.5, 0.10])


def test_tcp_target_lead_is_limited_from_measured_pose(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(
        driver.PiperRobotConfig(max_eef_target_lead_m=0.01, max_eef_target_lead_rad=None)
    )
    piper.connect()

    piper.send_action(
        {
            "ee.x": 0.45,
            "ee.y": -0.20,
            "ee.z": 0.40,
            "ee.roll": 0.10,
            "ee.pitch": -0.20,
            "ee.yaw": 0.30,
        }
    )

    tcp_target = FakeCartesianServo.instances[-1].steps[-1][2]
    assert np.linalg.norm(np.subtract(tcp_target[:3], arm.tcp_pose[:3])) == pytest.approx(0.01)


def test_tcp_rotation_uses_shortest_so3_limit_across_euler_wrap(monkeypatch):
    arm = FakeArm()
    arm.tcp_pose[3:] = [0.0, 0.0, math.pi - 0.05]
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(
        driver.PiperRobotConfig(max_eef_target_lead_m=None, max_eef_target_lead_rad=0.05)
    )
    piper.connect()

    piper.send_action(
        {
            "ee.x": arm.tcp_pose[0],
            "ee.y": arm.tcp_pose[1],
            "ee.z": arm.tcp_pose[2],
            "ee.roll": 0.0,
            "ee.pitch": 0.0,
            "ee.yaw": -math.pi + 0.05,
        }
    )

    tcp_target = FakeCartesianServo.instances[-1].steps[-1][2]
    rotation_delta = (
        Rotation.from_euler("xyz", tcp_target[3:]) * Rotation.from_euler("xyz", arm.tcp_pose[3:]).inv()
    )
    assert rotation_delta.magnitude() == pytest.approx(0.05)
    assert abs(tcp_target[5]) > 3.09


def test_tcp_action_rejects_unhealthy_arm_before_any_motion(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.healthy = False

    with pytest.raises(driver.PiperFeedbackError, match="unhealthy"):
        piper.send_action(
            {
                **dict.fromkeys(piper._EEF_KEYS, 0.0),
                "gripper.pos": 0.02,
            }
        )

    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_stale_tcp_feedback_rejects_cartesian_and_gripper_before_dispatch(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.events.clear()
    arm.effector.moves.clear()
    arm.set_tcp_timestamps(CURRENT_TIME_S - 1.0)

    with pytest.raises(driver.PiperFeedbackError, match="timestamp"):
        piper.send_action(
            {
                **dict(zip(piper._EEF_KEYS, arm.tcp_pose, strict=True)),
                "gripper.pos": 0.02,
            }
        )

    assert FakeCartesianServo.instances[-1].steps == []
    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_tcp_workspace_rejects_lead_limited_target_when_current_tcp_is_outside(monkeypatch):
    arm = FakeArm()
    arm.tcp_pose[0] = 0.70
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(
        driver.PiperRobotConfig(max_eef_target_lead_m=0.005, max_eef_target_lead_rad=None)
    )
    piper.connect()
    arm.events.clear()

    with pytest.raises(driver.PiperFeedbackError, match="current.*target.*workspace"):
        piper.send_action(
            {
                "ee.x": 0.65,
                "ee.y": arm.tcp_pose[1],
                "ee.z": arm.tcp_pose[2],
                "ee.roll": arm.tcp_pose[3],
                "ee.pitch": arm.tcp_pose[4],
                "ee.yaw": arm.tcp_pose[5],
            }
        )

    assert FakeCartesianServo.instances[-1].steps == []
    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_tcp_action_rejects_missing_gripper_before_servo(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    piper.gripper = None
    arm.events.clear()

    with pytest.raises(driver.PiperFeedbackError, match="gripper"):
        piper.send_action({**dict.fromkeys(piper._EEF_KEYS, 0.0), "gripper.pos": 0.02})

    assert FakeCartesianServo.instances[-1].steps == []
    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_tcp_action_reads_current_gripper_before_servo(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.events.clear()
    original_get_status = arm.effector.get_gripper_status

    def recording_get_status():
        arm.events.append(("read_gripper",))
        return original_get_status()

    arm.effector.get_gripper_status = recording_get_status
    servo = FakeCartesianServo.instances[-1]
    original_step = servo.step

    def recording_step(*args, **kwargs):
        arm.events.append(("servo_step",))
        return original_step(*args, **kwargs)

    servo.step = recording_step
    piper.send_action(dict.fromkeys(piper._EEF_KEYS, 0.0))

    read_index = next(index for index, event in enumerate(arm.events) if event[0] == "read_gripper")
    servo_index = next(index for index, event in enumerate(arm.events) if event[0] == "servo_step")
    move_index = next(index for index, event in enumerate(arm.events) if event[0] == "move_j")
    assert read_index < servo_index < move_index


def test_tcp_fk_exception_rejects_before_motion(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.fk_error = RuntimeError("FK failed")

    with pytest.raises(driver.PiperFeedbackError, match="Cartesian servo FK failed"):
        piper.send_action(dict.fromkeys(piper._EEF_KEYS, 0.0) | {"gripper.pos": 0.02})

    assert any(event[0] == "fk" for event in arm.events)
    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


@pytest.mark.parametrize(
    "flange2tcp_result",
    [
        [0.0] * 5,
        ["invalid", 0.0, 0.0, 0.0, 0.0, 0.0],
        [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_malformed_fk_tcp_conversion_result_rejects_before_motion(robot, flange2tcp_result):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.flange2tcp_result = flange2tcp_result

    with pytest.raises(driver.PiperFeedbackError, match="FK TCP pose"):
        piper.send_action(dict.fromkeys(piper._EEF_KEYS, 0.0) | {"gripper.pos": 0.02})

    assert any(event[0] == "get_flange2tcp_pose" for event in arm.events)
    assert not any(event[0] in {"move_j", "move_p"} for event in arm.events)
    assert arm.effector.moves == []


def test_partial_action_holds_missing_joints_and_clamps_relative_motion(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    measured_tcp = list(arm.tcp_pose)

    applied = piper.send_action({"joint_1.pos": 1.0, "joint_3.pos": -0.22, "gripper.pos": 0.2})

    assert arm.events[-1] == (
        "move_j",
        pytest.approx(
            [
                0.050003683069637546,
                0.10000736613927509,
                -0.21999875221388523,
                0.3000046451253053,
                0.39999455797206046,
                0.5000019241113355,
            ]
        ),
    )
    assert arm.effector.moves == [(0.07, 1.0)]
    assert set(applied) == set(piper.action_features)
    assert applied == pytest.approx(
        {
            "joint_1.pos": 0.050003683069637546,
            "joint_2.pos": 0.10000736613927509,
            "joint_3.pos": -0.21999875221388523,
            "joint_4.pos": 0.3000046451253053,
            "joint_5.pos": 0.39999455797206046,
            "joint_6.pos": 0.5000019241113355,
            **dict(zip(piper._EEF_KEYS, measured_tcp, strict=True)),
            "gripper.pos": 0.07,
        }
    )


def test_gripper_only_action_returns_complete_canonical_action(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    measured_joints = list(arm.joints)
    measured_tcp = list(arm.tcp_pose)

    applied = piper.send_action({"gripper.pos": 0.02})

    assert set(applied) == set(piper.action_features)
    assert [applied[key] for key in piper._JOINT_KEYS] == pytest.approx(measured_joints)
    assert [applied[key] for key in piper._EEF_KEYS] == pytest.approx(measured_tcp)
    assert applied["gripper.pos"] == pytest.approx(0.02)


def test_empty_action_returns_complete_measured_canonical_action(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    measured_joints = list(arm.joints)
    measured_tcp = list(arm.tcp_pose)

    applied = piper.send_action({})

    assert set(applied) == set(piper.action_features)
    assert [applied[key] for key in piper._JOINT_KEYS] == pytest.approx(measured_joints)
    assert [applied[key] for key in piper._EEF_KEYS] == pytest.approx(measured_tcp)
    assert applied["gripper.pos"] == pytest.approx(arm.effector.width_m)


@pytest.mark.parametrize("action", [{"joint_1.pos": 0.01}, {"gripper.pos": 0.02}])
@pytest.mark.parametrize(
    "tcp_pose",
    [
        [0.35, -0.10, 0.30],
        [0.35, -0.10, float("nan"), 0.10, -0.20, 0.30],
    ],
)
def test_joint_and_gripper_paths_reject_invalid_tcp_before_any_command(robot, action, tcp_pose):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.tcp_pose = tcp_pose

    with pytest.raises(driver.PiperFeedbackError, match="TCP pose feedback"):
        piper.send_action(action)

    assert not any(event[0] == "move_j" for event in arm.events)
    assert arm.effector.moves == []


def test_joint_action_rejects_missing_gripper_before_move_j(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    piper.gripper = None

    with pytest.raises(driver.PiperFeedbackError, match="gripper"):
        piper.send_action({"joint_1.pos": 0.01, "gripper.pos": 0.02})

    assert not any(event[0] == "move_j" for event in arm.events)
    assert arm.effector.moves == []


def test_joint_action_rejects_gripper_feedback_failure_before_move_j(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.effector.mode = "angle"

    with pytest.raises(driver.PiperFeedbackError, match="width mode"):
        piper.send_action({"joint_1.pos": 0.01})

    assert not any(event[0] == "move_j" for event in arm.events)
    assert arm.effector.moves == []


def test_requested_gripper_is_prepared_before_joint_motion_and_commanded_after(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.events.clear()
    original_get_tcp_pose = arm.get_tcp_pose
    original_move_gripper = arm.effector.move_gripper_m

    def recording_get_tcp_pose():
        arm.events.append(("get_tcp_pose",))
        return original_get_tcp_pose()

    def recording_move_gripper(*, value: float, force: float) -> None:
        arm.events.append(("move_gripper", value, force))
        original_move_gripper(value=value, force=force)

    arm.get_tcp_pose = recording_get_tcp_pose
    arm.effector.move_gripper_m = recording_move_gripper

    applied = piper.send_action({"joint_1.pos": 0.01, "gripper.pos": 0.20})

    tcp_index = next(index for index, event in enumerate(arm.events) if event[0] == "get_tcp_pose")
    joint_index = next(index for index, event in enumerate(arm.events) if event[0] == "move_j")
    gripper_index = next(index for index, event in enumerate(arm.events) if event[0] == "move_gripper")
    assert tcp_index < joint_index < gripper_index
    assert arm.events[gripper_index][1:] == pytest.approx((0.07, 1.0))
    assert applied["gripper.pos"] == pytest.approx(0.07)


def test_action_is_clamped_to_model_limits_and_returns_the_encoded_target(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig(max_relative_target=None))
    piper.connect()

    try:
        applied = piper.send_action({"joint_1.pos": 9.0})
    finally:
        piper.disconnect()

    assert arm.events[-2] == (
        "move_j",
        pytest.approx(
            [
                2.6179938779914944,
                0.10000736613927509,
                -0.19999727898603023,
                0.3000046451253053,
                0.39999455797206046,
                0.5000019241113355,
            ]
        ),
    )
    assert applied["joint_1.pos"] == pytest.approx(2.6179938779914944)


def test_action_fails_closed_when_measured_joint_is_outside_model_limits(monkeypatch):
    arm = FakeArm()
    arm.joints[2] = 0.1
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig())
    piper.connect()

    try:
        with pytest.raises(driver.PiperFeedbackError, match="joint_3.pos"):
            piper.send_action({"joint_1.pos": 0.01})
    finally:
        piper.disconnect()

    assert not any(event[0] == "move_j" for event in arm.events)


def test_action_accepts_measured_joint_within_default_zero_position_tolerance(monkeypatch):
    arm = FakeArm()
    arm.joints[2] = 0.020403
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig())
    piper.connect()

    try:
        applied = piper.send_action({"joint_1.pos": 0.01})
    finally:
        piper.disconnect()

    assert any(event[0] == "move_j" for event in arm.events)
    assert applied["joint_1.pos"] == pytest.approx(0.010000736613927507)


def test_action_returns_the_millidegree_target_encoded_by_sdk(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig(max_relative_target=None))
    piper.connect()

    try:
        applied = piper.send_action({"joint_1.pos": 0.123456})
    finally:
        piper.disconnect()

    assert applied["joint_1.pos"] == pytest.approx(0.12346459128607887)
    assert arm.events[-2][1][0] == pytest.approx(0.12346459128607887)


def test_invalid_gripper_value_rejects_whole_action_before_joint_command(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig())
    piper.connect()

    try:
        with pytest.raises(ValueError, match="gripper.pos"):
            piper.send_action({"joint_1.pos": 0.01, "gripper.pos": float("nan")})
    finally:
        piper.disconnect()

    assert not any(event[0] == "move_j" for event in arm.events)
    assert arm.effector.moves == []


def test_missing_or_wrong_mode_feedback_is_rejected(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()
    arm.get_joint_angles = lambda: None

    with pytest.raises(driver.PiperFeedbackError, match="joint angles"):
        piper.get_observation()

    arm.get_joint_angles = lambda: SimpleNamespace(msg=[0.0] * 6)
    arm.effector.mode = "angle"
    with pytest.raises(driver.PiperFeedbackError, match="width mode"):
        piper.get_observation()


def test_camera_timeout_preserves_current_robot_feedback(robot):
    piper, _arm, _sdk_configs = robot
    piper.cameras["wrist"] = TimeoutCamera()
    piper.connect()

    with pytest.raises(driver.PiperCameraTimeoutError) as exc_info:
        piper.get_observation()

    assert exc_info.value.camera_name == "wrist"
    assert exc_info.value.observation["joint_2.pos"] == pytest.approx(0.1)
    assert exc_info.value.observation["gripper.pos"] == pytest.approx(0.04)
    assert "wrist" not in exc_info.value.observation


def test_disconnect_releases_resources_without_disabling_arm_by_default(robot):
    piper, arm, _sdk_configs = robot
    camera = FakeCamera()
    piper.cameras["wrist"] = camera
    piper.connect()

    piper.disconnect()
    piper.disconnect()

    assert camera.is_connected is False
    assert ("disable",) not in arm.events
    assert arm.events.count(("disconnect",)) == 1
    assert piper.is_connected is False


def test_disconnect_closes_sdk_even_when_optional_disable_fails(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig(disable_on_disconnect=True))
    piper.connect()

    def fail_disable():
        arm.events.append(("disable",))
        raise RuntimeError("disable failed")

    arm.disable = fail_disable
    with pytest.raises(RuntimeError, match="disable failed"):
        piper.disconnect()

    assert ("disconnect",) in arm.events
    assert piper.is_connected is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"speed_percent": 101}, "speed_percent"),
        ({"max_relative_target": 0.0}, "max_relative_target"),
        ({"joint_limit_tolerance_rad": -0.01}, "joint_limit_tolerance_rad"),
        ({"gripper_force_n": 3.1}, "gripper_force_n"),
        ({"gripper_min_width_m": 0.08, "gripper_max_width_m": 0.07}, "gripper"),
    ],
)
def test_config_rejects_unsafe_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        driver.PiperRobotConfig(**kwargs)
