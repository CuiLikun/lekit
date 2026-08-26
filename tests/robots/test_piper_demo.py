from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from lekit.robots.piper import demo

CURRENT_TIME_S = 1_700_000_000.0


def message(value, *, hz: float = 100.0):
    return SimpleNamespace(msg=value, hz=hz, timestamp=CURRENT_TIME_S)


def healthy_foc_status(*, enabled: bool = False):
    return SimpleNamespace(
        voltage_too_low=False,
        motor_overheating=False,
        driver_overcurrent=False,
        driver_overheating=False,
        collision_status=False,
        sensor_status=False,
        driver_error_status=False,
        driver_enable_status=enabled,
        stall_status=False,
        homing_status=True,
    )


class FakeArm:
    joint_nums = 6

    def __init__(self) -> None:
        self.comm_error = False
        self.comm_error_after_checks: int | None = None
        self.comm_checks = 0
        self.enabled = False
        self.driver_fault_joint: int | None = None
        self.missing_driver_status_joint: int | None = None
        self.missing_driver_field_joint: int | None = None
        self.zero_hz_joint: int | None = None
        self.fps_readings = [200.0]
        self.motor_hz_readings = {index: [100.0] for index in range(1, 7)}
        self.driver_hz_readings = {index: [40.0] for index in range(1, 7)}
        self.software_version = "S-V1.8-2"
        self.firmware_error: Exception | None = None
        self.events: list[str] = []

    @staticmethod
    def _next(readings: list[float]) -> float:
        if len(readings) > 1:
            return readings.pop(0)
        return readings[0]

    def has_comm_error(self) -> bool:
        self.events.append("communication")
        self.comm_checks += 1
        if self.comm_error_after_checks is not None and self.comm_checks >= self.comm_error_after_checks:
            self.comm_error = True
        return self.comm_error

    def get_comm_error(self):
        return "CAN receive overflow" if self.comm_error else None

    def is_ok(self) -> bool:
        return not self.comm_error

    def get_fps(self) -> float:
        return self._next(self.fps_readings)

    def get_firmware(self):
        if self.firmware_error is not None:
            raise self.firmware_error
        return {
            "software_version": self.software_version,
            "hardware_version": "H-V1.2-1",
            "production_date": "250925",
            "node_type": "piper",
            "node_number": 1,
        }

    def get_arm_status(self):
        self.events.append("arm_status")
        return message(
            SimpleNamespace(
                ctrl_mode=1,
                arm_status=0,
                mode_feedback=1,
                teach_status=0,
                motion_status=0,
                trajectory_num=0,
                err_code=0,
                err_status=SimpleNamespace(),
            )
        )

    def get_flange_pose(self):
        return message([0.2, 0.0, 0.3, 0.0, 1.57, 0.0])

    def get_motor_states(self, joint_index: int):
        hz = 0.0 if joint_index == self.zero_hz_joint else self._next(self.motor_hz_readings[joint_index])
        return message(
            SimpleNamespace(position=0.1 * joint_index, velocity=0.0, current=0.2, torque=0.01),
            hz=hz,
        )

    def get_driver_states(self, joint_index: int):
        status = healthy_foc_status(enabled=self.enabled)
        if joint_index == self.driver_fault_joint:
            status.driver_overcurrent = True
        if joint_index == self.missing_driver_field_joint:
            del status.stall_status
        if joint_index == self.missing_driver_status_joint:
            status = None
        return message(
            SimpleNamespace(vol=24.0, foc_temp=31.0, motor_temp=29.0, bus_current=0.2, foc_status=status),
            hz=self._next(self.driver_hz_readings[joint_index]),
        )

    def get_joint_limits_enabled(self) -> bool:
        return True

    def enable(self) -> bool:
        self.events.append("enable")
        self.enabled = True
        return True

    def get_joints_enable_status_list(self) -> list[bool]:
        return [self.enabled] * 6


class FakeGripper:
    def __init__(self, width: float = 0.04) -> None:
        self.width = width
        self.ok = True
        self.mode = "width"
        self.homed = True
        self.fps_readings = [100.0]
        self.missing_foc_status = False
        self.missing_foc_field = False

    def is_ok(self) -> bool:
        return self.ok

    def get_fps(self) -> float:
        return FakeArm._next(self.fps_readings)

    def get_gripper_status(self):
        foc_status = healthy_foc_status(enabled=True)
        foc_status.homing_status = self.homed
        if self.missing_foc_field:
            del foc_status.sensor_status
        if self.missing_foc_status:
            foc_status = None
        return message(SimpleNamespace(value=self.width, force=1.0, mode=self.mode, foc_status=foc_status))


class FakeRobot:
    def __init__(self, config) -> None:
        self.config = config
        self.arm = FakeArm()
        self.gripper = FakeGripper() if config.include_gripper else None
        self.connected = False
        self.actions: list[dict[str, float]] = []
        self.fail_on_action: int | None = None
        self.interrupt_on_action: int | None = None
        self.stall_on_action: int | None = None
        self._joint_limits = {
            "joint_1.pos": (-2.617994, 2.617994),
            "joint_2.pos": (0.0, 3.141593),
            "joint_3.pos": (-2.967060, 0.0),
            "joint_4.pos": (-1.745330, 1.745330),
            "joint_5.pos": (-1.221730, 1.221730),
            "joint_6.pos": (-2.094396, 2.094396),
        }
        self.observation = {
            "joint_1.pos": 0.1,
            "joint_2.pos": 0.2,
            "joint_3.pos": -0.3,
            "joint_4.pos": 0.4,
            "joint_5.pos": 0.5,
            "joint_6.pos": -0.6,
        }
        if self.gripper is not None:
            self.observation["gripper.pos"] = self.gripper.width

    @property
    def is_connected(self) -> bool:
        return self.connected

    def __enter__(self):
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.connected = False

    def get_observation(self) -> dict[str, float]:
        return dict(self.observation)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.actions.append(dict(action))
        if self.interrupt_on_action == len(self.actions):
            raise KeyboardInterrupt
        if self.fail_on_action == len(self.actions):
            raise RuntimeError("simulated return failure")
        if self.stall_on_action == len(self.actions):
            return {**self.observation, **action}
        self.observation.update(action)
        if self.gripper is not None and "gripper.pos" in action:
            self.gripper.width = action["gripper.pos"]
        return dict(self.observation)


def make_harness(*, force_terminal: bool = False):
    robots: list[FakeRobot] = []
    output = StringIO()
    console = Console(file=output, force_terminal=force_terminal, color_system="standard", width=120)

    def create_robot(config):
        robot = FakeRobot(config)
        robots.append(robot)
        return robot

    return create_robot, robots, console, output


def test_static_self_test_renders_rich_report_without_moving():
    create_robot, robots, console, output = make_harness(force_terminal=True)

    exit_code = demo.main([], robot_factory=create_robot, console=console)

    robot = robots[0]
    rendered = output.getvalue()
    assert exit_code == 0
    assert robot.config.auto_enable is False
    assert robot.actions == []
    assert robot.connected is False
    assert "\x1b[" in rendered
    assert "Piper 完全自检" in rendered
    assert "通信与固件" in rendered
    assert "机械臂状态" in rendered
    assert "六轴电机与驱动器" in rendered
    assert "夹爪" in rendered
    assert "PASS" in rendered
    assert "动态自检未启用" in rendered


def test_static_failure_is_reported_and_returns_nonzero():
    create_robot, robots, console, output = make_harness()

    def create_faulty_robot(config):
        robot = create_robot(config)
        robot.arm.comm_error = True
        return robot

    exit_code = demo.main([], robot_factory=create_faulty_robot, console=console)

    assert exit_code == 1
    assert robots[0].actions == []
    assert "FAIL" in output.getvalue()
    assert "CAN receive overflow" in output.getvalue()


def test_static_self_test_waits_for_sdk_frequency_monitor_to_warm_up(monkeypatch):
    create_robot, robots, console, output = make_harness()
    monkeypatch.setattr(demo.time, "sleep", lambda _seconds: None)

    def create_warming_robot(config):
        robot = create_robot(config)
        robot.arm.fps_readings = [0.0, 2850.0]
        return robot

    exit_code = demo.main([], robot_factory=create_warming_robot, console=console)

    assert exit_code == 0
    assert robots[0].arm.fps_readings == [2850.0]
    assert "2850.0 Hz" in output.getvalue()


def test_communication_error_during_frequency_warmup_fails_static_check(monkeypatch):
    create_robot, robots, console, output = make_harness()
    monkeypatch.setattr(demo.time, "sleep", lambda _seconds: None)

    def create_late_error_robot(config):
        robot = create_robot(config)
        robot.arm.fps_readings = [0.0, 2850.0]
        robot.arm.comm_error_after_checks = 2
        return robot

    exit_code = demo.main([], robot_factory=create_late_error_robot, console=console)

    assert exit_code == 1
    assert robots[0].actions == []
    assert "CAN receive overflow" in output.getvalue()


def test_static_self_test_waits_for_joint_and_gripper_frequencies_to_warm_up(monkeypatch):
    create_robot, robots, console, output = make_harness()
    monkeypatch.setattr(demo.time, "sleep", lambda _seconds: None)

    def create_warming_robot(config):
        robot = create_robot(config)
        robot.arm.motor_hz_readings[1] = [0.0, 200.0]
        robot.arm.driver_hz_readings[1] = [0.0, 40.0]
        assert robot.gripper is not None
        robot.gripper.fps_readings = [0.0, 200.0]
        return robot

    exit_code = demo.main([], robot_factory=create_warming_robot, console=console)

    assert exit_code == 0
    assert robots[0].arm.motor_hz_readings[1] == [200.0]
    assert robots[0].arm.driver_hz_readings[1] == [40.0]
    assert robots[0].gripper is not None
    assert robots[0].gripper.fps_readings == [200.0]


def test_firmware_driver_mismatch_blocks_motion_and_reports_expected_selection():
    create_robot, robots, console, output = make_harness()

    def create_v188_robot(config):
        robot = create_robot(config)
        robot.arm.software_version = "S-V1.8-8"
        return robot

    exit_code = demo.main([], robot_factory=create_v188_robot, console=console)

    assert exit_code == 1
    assert robots[0].actions == []
    assert "--firmware-version v188" in output.getvalue()


@pytest.mark.parametrize("software_version", ["S-V2.0-1", "unknown", ""])
def test_unrecognized_firmware_fails_closed(software_version):
    create_robot, robots, console, output = make_harness()

    def create_unknown_firmware_robot(config):
        robot = create_robot(config)
        robot.arm.software_version = software_version
        return robot

    exit_code = demo.main([], robot_factory=create_unknown_firmware_robot, console=console)

    assert exit_code == 1
    assert robots[0].actions == []
    assert "无法确定兼容驱动" in output.getvalue()


def test_firmware_query_exception_blocks_confirmation_and_motion():
    create_robot, robots, console, output = make_harness()

    def create_firmware_failure_robot(config):
        robot = create_robot(config)
        robot.arm.firmware_error = TimeoutError("firmware query timed out")
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_firmware_failure_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    assert exit_code == 1
    assert robots[0].arm.enabled is False
    assert robots[0].actions == []
    assert "firmware query timed out" in output.getvalue()
    assert "静态安全检查失败" in output.getvalue()


def test_full_self_test_moves_every_joint_and_gripper_then_restores_start_state():
    create_robot, robots, console, output = make_harness()

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    robot = robots[0]
    assert exit_code == 0
    assert robot.config.auto_enable is False
    assert robot.arm.events.index("arm_status") < robot.arm.events.index("enable")
    assert robot.observation == {
        "joint_1.pos": pytest.approx(0.1),
        "joint_2.pos": pytest.approx(0.2),
        "joint_3.pos": pytest.approx(-0.3),
        "joint_4.pos": pytest.approx(0.4),
        "joint_5.pos": pytest.approx(0.5),
        "joint_6.pos": pytest.approx(-0.6),
        "gripper.pos": pytest.approx(0.04),
    }
    assert len(robot.actions) == 14
    for joint_index in range(1, 7):
        forward, restored = robot.actions[(joint_index - 1) * 2 : joint_index * 2]
        key = f"joint_{joint_index}.pos"
        assert forward == {key: pytest.approx(robot.observation[key] + 0.01)}
        assert restored == {key: pytest.approx(robot.observation[key])}
    assert robot.actions[-2:] == [
        {"gripper.pos": pytest.approx(0.045)},
        {"gripper.pos": pytest.approx(0.04)},
    ]
    assert "动态运动检查" in output.getvalue()
    assert "全部关键检查通过" in output.getvalue()


def test_full_self_test_cancelled_by_operator_never_enables_or_moves():
    create_robot, robots, console, output = make_harness()

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_robot,
        console=console,
        confirm_motion=lambda: False,
    )

    assert exit_code == 0
    assert robots[0].arm.enabled is False
    assert robots[0].actions == []
    assert "用户取消" in output.getvalue()


@pytest.mark.parametrize(
    "invalid_state",
    ["unhealthy", "zero_fps", "wrong_mode", "non_finite_width", "out_of_range_width"],
)
def test_gripper_state_regression_after_confirmation_blocks_enable_and_motion(invalid_state: str):
    create_robot, robots, console, output = make_harness()

    def invalidate_gripper_after_static_checks() -> bool:
        gripper = robots[0].gripper
        assert gripper is not None
        gripper.homed = False
        if invalid_state == "unhealthy":
            gripper.ok = False
        elif invalid_state == "zero_fps":
            gripper.fps_readings = [0.0]
        elif invalid_state == "wrong_mode":
            gripper.mode = "angle"
        elif invalid_state == "non_finite_width":
            gripper.width = float("nan")
        else:
            gripper.width = 0.08
        return True

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_robot,
        console=console,
        confirm_motion=invalidate_gripper_after_static_checks,
    )

    robot = robots[0]
    assert exit_code == 1
    assert robot.arm.enabled is False
    assert robot.actions == []
    assert "确认后安全复核" in output.getvalue()


def test_keyboard_interrupt_at_confirmation_is_reported_without_motion():
    create_robot, robots, console, output = make_harness()

    def interrupt_confirmation() -> bool:
        raise KeyboardInterrupt

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_robot,
        console=console,
        confirm_motion=interrupt_confirmation,
    )

    assert exit_code == 1
    assert robots[0].arm.enabled is False
    assert robots[0].actions == []
    assert "用户中断" in output.getvalue()


def test_full_self_test_revalidates_safety_state_after_confirmation():
    create_robot, robots, console, output = make_harness()

    def introduce_fault_then_confirm() -> bool:
        robots[0].arm.driver_fault_joint = 2
        return True

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_robot,
        console=console,
        confirm_motion=introduce_fault_then_confirm,
    )

    assert exit_code == 1
    assert robots[0].arm.enabled is False
    assert robots[0].actions == []
    assert "确认后安全复核" in output.getvalue()
    assert "driver_overcurrent" in output.getvalue()


def test_critical_static_failure_blocks_confirmation_and_all_motion():
    create_robot, robots, console, output = make_harness()

    def create_faulty_robot(config):
        robot = create_robot(config)
        robot.arm.driver_fault_joint = 3
        return robot

    def unexpected_confirmation() -> bool:
        raise AssertionError("confirmation must not run after a safety failure")

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_faulty_robot,
        console=console,
        confirm_motion=unexpected_confirmation,
    )

    assert exit_code == 1
    assert robots[0].arm.enabled is False
    assert robots[0].actions == []
    assert "driver_overcurrent" in output.getvalue()
    assert "静态安全检查失败" in output.getvalue()


def test_out_of_limit_start_blocks_round_trip_before_enabling_or_moving():
    create_robot, robots, console, output = make_harness()

    def create_out_of_limit_robot(config):
        robot = create_robot(config)
        robot.observation["joint_3.pos"] = 0.009
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_out_of_limit_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    assert exit_code == 1
    assert robots[0].arm.enabled is False
    assert robots[0].actions == []
    assert "joint_3.pos" in output.getvalue()
    assert "无法安全原位往返" in output.getvalue()


def test_missing_per_joint_feedback_frequency_fails_static_self_test(monkeypatch):
    create_robot, robots, console, output = make_harness()
    monkeypatch.setattr(demo, "FEEDBACK_WARMUP_S", 0.0)

    def create_stale_robot(config):
        robot = create_robot(config)
        robot.arm.zero_hz_joint = 4
        return robot

    exit_code = demo.main([], robot_factory=create_stale_robot, console=console)

    assert exit_code == 1
    assert robots[0].actions == []
    assert "关节 4" in output.getvalue()
    assert "反馈频率" in output.getvalue()


@pytest.mark.parametrize("missing_kind", ["status", "field"])
def test_incomplete_driver_fault_feedback_fails_closed(missing_kind):
    create_robot, robots, console, output = make_harness()

    def create_incomplete_robot(config):
        robot = create_robot(config)
        if missing_kind == "status":
            robot.arm.missing_driver_status_joint = 2
        else:
            robot.arm.missing_driver_field_joint = 2
        return robot

    exit_code = demo.main([], robot_factory=create_incomplete_robot, console=console)

    assert exit_code == 1
    assert robots[0].actions == []
    assert "关节 2" in output.getvalue()
    assert "故障状态反馈不完整" in output.getvalue()


@pytest.mark.parametrize("missing_kind", ["status", "field"])
def test_incomplete_gripper_fault_feedback_fails_closed(missing_kind):
    create_robot, robots, console, output = make_harness()

    def create_incomplete_robot(config):
        robot = create_robot(config)
        assert robot.gripper is not None
        if missing_kind == "status":
            robot.gripper.missing_foc_status = True
        else:
            robot.gripper.missing_foc_field = True
        return robot

    exit_code = demo.main([], robot_factory=create_incomplete_robot, console=console)

    assert exit_code == 1
    assert robots[0].actions == []
    assert "AGX 夹爪" in output.getvalue()
    assert "故障状态反馈不完整" in output.getvalue()


def test_joint_return_failure_aborts_remaining_checks_and_attempts_full_restore():
    create_robot, robots, console, output = make_harness()

    def create_return_failure_robot(config):
        robot = create_robot(config)
        robot.fail_on_action = 2
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_return_failure_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    robot = robots[0]
    assert exit_code == 1
    assert len(robot.actions) == 3
    assert robot.actions[0] == {"joint_1.pos": pytest.approx(0.11)}
    assert robot.actions[1] == {"joint_1.pos": pytest.approx(0.1)}
    assert set(robot.actions[2]) == {f"joint_{index}.pos" for index in range(1, 7)}
    assert robot.observation["joint_1.pos"] == pytest.approx(0.1)
    assert "中止剩余动态检查" in output.getvalue()


def test_keyboard_interrupt_during_motion_is_reported_and_attempts_restore():
    create_robot, robots, console, output = make_harness()

    def create_interrupted_robot(config):
        robot = create_robot(config)
        robot.interrupt_on_action = 1
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_interrupted_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    robot = robots[0]
    assert exit_code == 1
    assert len(robot.actions) == 2
    assert set(robot.actions[1]) == {f"joint_{index}.pos" for index in range(1, 7)}
    assert robot.observation["joint_1.pos"] == pytest.approx(0.1)
    assert "用户中断" in output.getvalue()
    assert "故障恢复" in output.getvalue()


def test_accepted_command_with_stalled_feedback_times_out_and_restores(monkeypatch):
    create_robot, robots, console, output = make_harness()
    monkeypatch.setattr(demo, "MOTION_TIMEOUT_S", 0.0)

    def create_stalled_robot(config):
        robot = create_robot(config)
        robot.stall_on_action = 1
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_stalled_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    robot = robots[0]
    assert exit_code == 1
    assert len(robot.actions) == 2
    assert robot.actions[0] == {"joint_1.pos": pytest.approx(0.11)}
    assert set(robot.actions[1]) == {f"joint_{index}.pos" for index in range(1, 7)}
    assert "未到达目标" in output.getvalue()
    assert "中止剩余动态检查" in output.getvalue()
    assert "故障恢复" in output.getvalue()


def test_gripper_motion_failure_verifies_and_reports_restore(monkeypatch):
    create_robot, robots, console, output = make_harness()
    monkeypatch.setattr(demo, "MOTION_TIMEOUT_S", 0.0)

    def create_stalled_gripper_robot(config):
        robot = create_robot(config)
        robot.stall_on_action = 13
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_stalled_gripper_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    robot = robots[0]
    assert exit_code == 1
    assert len(robot.actions) == 14
    assert robot.observation["gripper.pos"] == pytest.approx(0.04)
    assert "夹爪故障恢复" in output.getvalue()
    assert "已恢复夹爪初始宽度" in output.getvalue()


def test_false_gripper_homing_status_warns_but_does_not_skip_safe_motion_test():
    create_robot, robots, console, output = make_harness()

    def create_unhomed_robot(config):
        robot = create_robot(config)
        assert robot.gripper is not None
        robot.gripper.homed = False
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_unhomed_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    robot = robots[0]
    assert exit_code == 0
    assert len(robot.actions) == 14
    assert robot.actions[-2:] == [
        {"gripper.pos": pytest.approx(0.045)},
        {"gripper.pos": pytest.approx(0.04)},
    ]
    assert "零位状态位未置位" in output.getvalue()
    assert "夹爪往返" in output.getvalue()


def test_missing_gripper_motion_baseline_is_reported_as_gripper_failure():
    create_robot, robots, console, output = make_harness()

    def create_missing_baseline_robot(config):
        robot = create_robot(config)
        del robot.observation["gripper.pos"]
        return robot

    exit_code = demo.main(
        ["--full"],
        robot_factory=create_missing_baseline_robot,
        console=console,
        confirm_motion=lambda: True,
    )

    assert exit_code == 1
    assert len(robots[0].actions) == 12
    assert "夹爪往返" in output.getvalue()
    assert "gripper.pos" in output.getvalue()
