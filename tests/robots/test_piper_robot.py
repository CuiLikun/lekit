from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lekit.robots.piper import piper_robot as driver


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
        return SimpleNamespace(msg=message, hz=100.0, timestamp=1.0)

    def move_gripper_m(self, *, value: float, force: float) -> None:
        self.width_m = value
        self.moves.append((value, force))


class FakeArm:
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="agx_gripper"))

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.connected = False
        self.joints = [0.0, 0.1, -0.2, 0.3, 0.4, 0.5]
        self.effector = FakeEffector()
        self.joint_limits_enabled = False
        self._parser = SimpleNamespace(joint_12=object(), joint_34=object(), joint_56=object())

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

    def enable(self) -> bool:
        self.events.append(("enable",))
        return True

    def disable(self) -> bool:
        self.events.append(("disable",))
        return True

    def set_speed_percent(self, percent: int) -> None:
        self.events.append(("set_speed_percent", percent))

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
        return SimpleNamespace(msg=list(self.joints), hz=200.0, timestamp=1.0)

    def move_j(self, joints: list[float]) -> None:
        self.joints = list(joints)
        self.events.append(("move_j", list(joints)))


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
    assert robot.action_features == expected
    assert robot.observation_features == {**expected, "wrist": (6, 8, 3)}


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
        ("set_joint_limits_enabled", True),
        ("set_speed_percent", 20),
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
    arm._parser = SimpleNamespace(joint_12=None, joint_34=None, joint_56=None)
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
    monkeypatch.setattr(driver, "create_piper_sdk_arm", lambda _config: arm)
    piper = driver.PiperRobot(driver.PiperRobotConfig(enable_poll_interval_s=0.001))

    try:
        piper.connect()
    finally:
        piper.disconnect()

    assert joint_reads == 3
    assert gripper_reads == 2


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


def test_partial_action_holds_missing_joints_and_clamps_relative_motion(robot):
    piper, arm, _sdk_configs = robot
    piper.connect()

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
    assert applied == pytest.approx(
        {
            "joint_1.pos": 0.050003683069637546,
            "joint_2.pos": 0.10000736613927509,
            "joint_3.pos": -0.21999875221388523,
            "joint_4.pos": 0.3000046451253053,
            "joint_5.pos": 0.39999455797206046,
            "joint_6.pos": 0.5000019241113355,
            "gripper.pos": 0.07,
        }
    )


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
