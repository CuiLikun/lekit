from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from robots.jaka_robot import jaka_robot as driver


class FakeRC:
    def __init__(self):
        self.calls: list[tuple] = []
        self.joints = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        self.tcp_mm = (100.0, 200.0, 300.0, 0.1, 0.2, 0.3)
        self.powered_on = False
        self.enabled = False
        self.servo_enabled = False
        self.joint_reads = 0
        self.tcp_reads = 0

    def _call(self, name, *args):
        self.calls.append((name, *args))
        return (0,)

    def login(self, **kwargs):
        return self._call("login", **kwargs)

    def logout(self):
        return self._call("logout")

    def power_on(self):
        self.powered_on = True
        return self._call("power_on")

    def power_off(self):
        self.powered_on = False
        return self._call("power_off")

    def enable_robot(self):
        self.enabled = True
        return self._call("enable_robot")

    def disable_robot(self):
        self.enabled = False
        return self._call("disable_robot")

    def get_robot_status_simple(self):
        return (0, 0, "", self.powered_on, self.enabled)

    def set_tool_id(self, value):
        return self._call("set_tool_id", value)

    def set_user_frame_id(self, value):
        return self._call("set_user_frame_id", value)

    def set_collision_level(self, value):
        return self._call("set_collision_level", value)

    def set_motion_planner(self, value):
        return self._call("set_motion_planner", value)

    def set_vibsuppress_mode(self, value):
        return self._call("set_vibsuppress_mode", value)

    def vibsuppress_on(self, frequency):
        return self._call("vibsuppress_on", frequency)

    def servo_move_use_joint_LPF(self, cutoff):  # noqa: N802
        return self._call("servo_move_use_joint_LPF", cutoff)

    def servo_move_use_joint_NLF(self, max_vr, max_ar, max_jr):  # noqa: N802
        return self._call("servo_move_use_joint_NLF", max_vr, max_ar, max_jr)

    def servo_move_use_carte_NLF(self, max_vp, max_ap, max_jp, max_vr, max_ar, max_jr):  # noqa: N802
        return self._call("servo_move_use_carte_NLF", max_vp, max_ap, max_jp, max_vr, max_ar, max_jr)

    def servo_move_enable(self, value, _block=True):
        self.servo_enabled = bool(value)
        return self._call("servo_move_enable", value)

    def is_in_servomove(self):
        return (0, self.servo_enabled)

    def get_actual_joint_position(self):
        self.joint_reads += 1
        return (0, self.joints)

    def get_actual_tcp_position(self):
        self.tcp_reads += 1
        return (0, self.tcp_mm)

    def servo_j(self, joints, mode, step):
        self.calls.append(("servo_j", joints, mode, step))
        return (0, 2)

    def servo_p(self, pose, mode, step):
        self.calls.append(("servo_p", pose, mode, step))
        return (0, 3)

    def joint_move(self, joints, mode, is_block, speed):
        return self._call("joint_move", joints, mode, is_block, speed)

    def linear_move(self, pose, mode, is_block, speed):
        return self._call("linear_move", pose, mode, is_block, speed)

    def set_analog_output(self, *, iotype, index, value):
        self.calls.append(("set_analog_output", iotype, index, value))
        return (0,)


def wait_until(predicate, *, timeout: float = 0.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not met before timeout")


def servo_calls(rc: FakeRC, operation: str) -> list[tuple]:
    return [call for call in rc.calls if call[0] == operation]


def test_servo_deadline_wait_reserves_a_short_precision_window():
    now = [0.0]
    waits: list[float] = []

    def clock() -> float:
        now[0] += 0.0001
        return now[0]

    class FakeStop:
        def wait(self, duration: float) -> bool:
            waits.append(duration)
            now[0] += duration
            return False

        def is_set(self) -> bool:
            return False

    completed = driver._wait_for_servo_deadline(
        0.008,
        stop=FakeStop(),
        clock=clock,
        spin_threshold_s=0.0005,
    )

    assert completed is True
    assert waits == pytest.approx([0.0074])
    assert now[0] >= 0.008


@pytest.fixture
def robot(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    robot = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2"))
    robot.connect()
    yield robot, rc
    robot.disconnect()


def test_joint_action_is_servo_bounded_and_observation_uses_si_units(robot):
    arm, rc = robot
    observation = arm.get_observation()
    assert observation["joint_2.pos"] == 0.1
    assert observation["ee.z"] == 0.3

    before = len(servo_calls(rc, "servo_j"))
    applied = arm.send_action({"joint_1.pos": 1.0, "gripper.pos": 0.4})
    assert applied["joint_1.pos"] == pytest.approx(0.05)
    assert applied["ee.x"] == pytest.approx(0.1)
    assert set(applied) == set(arm.action_features)
    wait_until(
        lambda: len(servo_calls(rc, "servo_j")) > before and servo_calls(rc, "servo_j")[-1][1][0] > 0.0
    )
    first_target = next(call for call in servo_calls(rc, "servo_j")[before:] if call[1][0] > 0.0)
    assert 0.0 < first_target[1][0] < 0.05
    assert first_target[2:] == (driver.ABS, 1)

    before = len(servo_calls(rc, "servo_j"))
    applied = arm.send_relative_action({"joint_2.pos": -0.02})
    assert applied["joint_2.pos"] == pytest.approx(0.08)
    wait_until(
        lambda: len(servo_calls(rc, "servo_j")) > before and servo_calls(rc, "servo_j")[-1][1][1] < 0.1
    )
    assert servo_calls(rc, "servo_j")[-1][1][1] < 0.1

    assert arm.send_relative_action({"gripper.pos": 0.1})["gripper.pos"] == pytest.approx(0.5)
    assert arm.send_relative_action({"gripper.pos": 0.1})["gripper.pos"] == pytest.approx(0.6)


def test_cartesian_action_is_servo_bounded_and_converted_to_mm(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2", max_eef_step_m=0.01))
    arm.connect()
    try:
        before = len(servo_calls(rc, "servo_p"))
        applied = arm.send_action({"ee.x": 1.0, "ee.roll": 1.0})
        assert applied["ee.x"] == pytest.approx(0.11)
        assert applied["ee.roll"] == pytest.approx(0.18)
        assert applied["joint_2.pos"] == pytest.approx(0.1)
        assert set(applied) == set(arm.action_features)
        wait_until(
            lambda: len(servo_calls(rc, "servo_p")) > before and servo_calls(rc, "servo_p")[-1][1][0] > 100.0
        )
        first_target = next(call for call in servo_calls(rc, "servo_p")[before:] if call[1][0] > 100.0)
        assert 100.0 < first_target[1][0] < 110.0
        assert 0.1 < first_target[1][3] < 0.18
        assert first_target[2:] == (driver.ABS, 1)

        applied = arm.send_relative_action({"ee.x": 0.005})
        assert applied["ee.x"] == pytest.approx(0.105)
    finally:
        arm.disconnect()


def test_cartesian_servo_filter_uses_sdk_units(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    config = driver.JakaRobotConfig(
        ip="10.0.0.2",
        servo_filter_mode="cartesian_nlf",
        servo_filter_eef_max_jerk_m_s3=1.5,
        servo_filter_eef_max_angular_jerk_rad_s3=2.5,
    )
    arm = driver.JakaRobot(config)
    arm.connect()
    try:
        calls = servo_calls(rc, "servo_move_use_carte_NLF")
        assert calls == [
            (
                "servo_move_use_carte_NLF",
                50.0,
                200.0,
                1500.0,
                math.degrees(0.5),
                math.degrees(1.0),
                math.degrees(2.5),
            )
        ]
    finally:
        arm.disconnect()


def test_cartesian_nlf_is_the_only_eef_trajectory_filter():
    arm = driver.JakaRobot(
        driver.JakaRobotConfig(
            ip="10.0.0.2",
            servo_filter_mode="cartesian_nlf",
            servo_eef_max_velocity_m_s=0.15,
            servo_eef_max_acceleration_m_s2=0.8,
        )
    )
    initial = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    target = np.array([0.13, 0.2, 0.3, 0.2, 0.2, 0.3])
    arm._servo_commanded_position = initial.copy()
    arm._servo_target = target.copy()
    arm._servo_linear_velocity.fill(0.0)
    arm._servo_angular_speed = 0.0

    frame = arm._step_servo_eef(driver.SERVO_CYCLE_S)

    # The controller-side NLF already enforces velocity, acceleration, and jerk.
    # Host interpolation here would apply those limits twice and add large lag.
    assert frame == pytest.approx(target)


def test_recent_observation_is_reused_by_eef_action(robot):
    arm, rc = robot
    arm.get_observation()
    before = (rc.joint_reads, rc.tcp_reads)

    applied = arm.send_action({"ee.x": 0.105})

    assert applied["ee.x"] == pytest.approx(0.105)
    assert (rc.joint_reads, rc.tcp_reads) == before


def test_servo_status_exposes_target_and_commanded_position(robot):
    arm, rc = robot

    status = arm.get_servo_status()

    assert status["filter_mode"] == "none"
    assert status["target"] == pytest.approx(rc.joints)
    assert status["commanded_position"] == pytest.approx(rc.joints)


def test_eef_orientation_limit_handles_rpy_wrap(monkeypatch):
    rc = FakeRC()
    rc.tcp_mm = (100.0, 200.0, 300.0, 0.1, 0.2, -3.13)
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2"))
    arm.connect()
    try:
        arm.get_observation()
        applied = arm.send_action({"ee.yaw": 3.14})
        assert applied["ee.yaw"] == pytest.approx(3.14)
    finally:
        arm.disconnect()


def test_feedback_connection_keeps_state_reads_off_servo_handle(monkeypatch):
    control_rc, feedback_rc = FakeRC(), FakeRC()
    handles = iter((control_rc, feedback_rc))
    monkeypatch.setattr(driver, "create_rc", lambda _ip: next(handles))
    arm = driver.JakaRobot(
        driver.JakaRobotConfig(
            ip="10.0.0.2",
            auto_enable_servo=False,
            separate_feedback_connection=True,
        )
    )
    arm.connect()
    try:
        arm.get_observation()
        assert control_rc.joint_reads == 0
        assert control_rc.tcp_reads == 0
        assert feedback_rc.joint_reads == 1
        assert feedback_rc.tcp_reads == 1
    finally:
        arm.disconnect()


def test_controller_status_reads_use_feedback_connection(monkeypatch):
    control_rc, feedback_rc = FakeRC(), FakeRC()
    handles = iter((control_rc, feedback_rc))
    monkeypatch.setattr(driver, "create_rc", lambda _ip: next(handles))
    arm = driver.JakaRobot(
        driver.JakaRobotConfig(
            ip="10.0.0.2",
            auto_enable_servo=False,
            separate_feedback_connection=True,
        )
    )
    arm.connect()
    status_thread: threading.Thread | None = None
    try:
        results: list[dict[str, bool] | bool] = []
        with arm._sdk_lock:
            status_thread = threading.Thread(
                target=lambda: results.extend((arm.get_controller_state(), arm.is_in_servo()))
            )
            status_thread.start()
            status_thread.join(timeout=0.1)
            assert not status_thread.is_alive()

        assert results == [
            {"powered_on": feedback_rc.powered_on, "enabled": feedback_rc.enabled},
            feedback_rc.servo_enabled,
        ]
    finally:
        if status_thread is not None:
            status_thread.join(timeout=0.5)
        arm.disconnect()


def test_blocking_servo_p_does_not_block_target_updates(monkeypatch):
    rc = FakeRC()
    servo_p_started = threading.Event()
    release_servo_p = threading.Event()

    def blocking_servo_p(pose, mode, step):
        servo_p_started.set()
        release_servo_p.wait(timeout=1.0)
        rc.calls.append(("servo_p", pose, mode, step))
        return (0, 1)

    rc.servo_p = blocking_servo_p
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2"))
    arm.connect()
    update_thread: threading.Thread | None = None
    try:
        arm.get_observation()
        arm.send_action({"ee.x": 0.105})
        assert servo_p_started.wait(timeout=0.5)

        update_thread = threading.Thread(target=lambda: arm.send_action({"ee.x": 0.106}))
        update_thread.start()
        update_thread.join(timeout=0.1)
        assert not update_thread.is_alive()
    finally:
        release_servo_p.set()
        if update_thread is not None:
            update_thread.join(timeout=0.5)
        arm.disconnect()


def test_action_representation_can_switch_and_mixed_frames_are_rejected(robot):
    arm, rc = robot

    arm.send_action({"joint_1.pos": 0.01})
    arm.send_action({"ee.x": 0.105})
    wait_until(lambda: bool(servo_calls(rc, "servo_p")))

    servo_call_count = len([call for call in rc.calls if call[0] in {"servo_j", "servo_p"}])
    with pytest.raises(ValueError, match="cannot mix joint and EEF"):
        arm.send_action({"joint_1.pos": 0.0, "ee.x": 0.1})
    assert len([call for call in rc.calls if call[0] in {"servo_j", "servo_p"}]) == servo_call_count


def test_gripper_only_action_does_not_send_an_arm_frame(robot):
    arm, rc = robot
    servo_call_count = len([call for call in rc.calls if call[0] in {"servo_j", "servo_p"}])

    applied = arm.send_action({"gripper.pos": 0.75})

    assert applied["gripper.pos"] == pytest.approx(0.75)
    assert set(applied) == set(arm.action_features)
    assert len([call for call in rc.calls if call[0] in {"servo_j", "servo_p"}]) == servo_call_count


@pytest.mark.parametrize(
    ("position", "expected_width"),
    [(0.0, 0.0), (0.7, 700.0), (1.0, 1000.0)],
)
def test_gripper_uses_extension_analog_output_channel_three(robot, position, expected_width):
    arm, rc = robot

    assert arm.set_gripper_position(position) == pytest.approx(position)

    assert rc.calls[-1] == (
        "set_analog_output",
        driver.IO_EXTEND,
        3,
        expected_width,
    )


def test_reconnect_preserves_an_enabled_controller_by_default(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)

    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2", auto_enable_servo=False))
    arm.connect()
    arm.disconnect()
    arm.connect()
    arm.disconnect()

    operations = [call[0] for call in rc.calls]
    assert operations.count("power_on") == 1
    assert operations.count("enable_robot") == 1
    assert "disable_robot" not in operations
    assert rc.powered_on is True
    assert rc.enabled is True


def test_non_servo_actions_use_controller_planned_moves(robot):
    arm, rc = robot
    arm.servo_enable(False)

    applied = arm.send_action({"joint_1.pos": 1.0}, use_servo=False)
    assert applied["joint_1.pos"] == pytest.approx(0.05)
    assert rc.calls[-1] == ("joint_move", (0.05, 0.1, 0.2, 0.3, 0.4, 0.5), driver.ABS, True, 0.5)

    applied = arm.send_relative_action({"ee.x": 0.02}, use_servo=False)
    assert applied["ee.x"] == pytest.approx(0.11)
    assert rc.calls[-1] == ("linear_move", (110.0, 200.0, 300.0, 0.1, 0.2, 0.3), driver.ABS, True, 100.0)

    arm.servo_enable(True)
    with pytest.raises(RuntimeError, match="Exit Servo Move"):
        arm.send_action({"joint_1.pos": 0.0}, use_servo=False)


def test_failed_sdk_result_has_operation_and_code(monkeypatch):
    rc = FakeRC()
    rc.set_collision_level = lambda _value: (-7,)
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2"))
    with pytest.raises(driver.JakaError, match="set_collision_level failed \\(-7\\)"):
        arm.connect()
    assert arm.rc is None


def test_config_normalizes_scalar_safety_limit_and_rejects_incomplete_joint_mapping():
    assert driver.JakaRobotConfig(max_relative_target=1).max_relative_target == 1.0
    with pytest.raises(ValueError, match="joint mapping"):
        driver.JakaRobotConfig(max_relative_target={"joint_1.pos": 0.1})


def test_config_validates_and_normalizes_reset_joints():
    config = driver.JakaRobotConfig(reset_joints=[0, 1, 2, 3, 4, 5])
    assert config.reset_joints == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    with pytest.raises(ValueError, match="exactly six"):
        driver.JakaRobotConfig(reset_joints=[0.0] * 5)
    with pytest.raises(ValueError, match="finite"):
        driver.JakaRobotConfig(reset_joints=[0.0, 1.0, 2.0, 3.0, 4.0, math.nan])


def test_configured_joint_and_eef_position_limits_are_enforced(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(
        driver.JakaRobotConfig(
            ip="10.0.0.2",
            joint_position_limits={"joint_1.pos": (-0.01, 0.01)},
        )
    )
    arm.connect()
    try:
        before = len(servo_calls(rc, "servo_j"))
        applied = arm.send_relative_action({"joint_1.pos": 0.05})
        assert applied["joint_1.pos"] == pytest.approx(0.01)
        wait_until(
            lambda: len(servo_calls(rc, "servo_j")) > before and servo_calls(rc, "servo_j")[-1][1][0] > 0.0
        )
        assert 0.0 < servo_calls(rc, "servo_j")[-1][1][0] <= 0.01
    finally:
        arm.disconnect()

    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(
        driver.JakaRobotConfig(
            ip="10.0.0.2",
            eef_pose_limits={"ee.x": (0.0, 0.103)},
        )
    )
    arm.connect()
    try:
        before = len(servo_calls(rc, "servo_p"))
        applied = arm.send_relative_action({"ee.x": 0.01})
        assert applied["ee.x"] == pytest.approx(0.103)
        wait_until(
            lambda: len(servo_calls(rc, "servo_p")) > before and servo_calls(rc, "servo_p")[-1][1][0] > 100.0
        )
        assert 100.0 < servo_calls(rc, "servo_p")[-1][1][0] <= 103.0
    finally:
        arm.disconnect()


def test_servo_sender_continues_at_eight_ms_and_stops_before_controller_exit(robot):
    arm, rc = robot
    wait_until(lambda: len(servo_calls(rc, "servo_j")) >= 4)
    count = len(servo_calls(rc, "servo_j"))
    time.sleep(0.025)
    assert len(servo_calls(rc, "servo_j")) > count

    arm.servo_enable(False)
    stopped_count = len(servo_calls(rc, "servo_j"))
    time.sleep(0.025)
    assert len(servo_calls(rc, "servo_j")) == stopped_count
    assert rc.calls[-1] == ("servo_move_enable", False)


def test_servo_enable_waits_for_the_first_controller_frame(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2", auto_enable_servo=False))
    arm.connect()
    try:
        rc.servo_j = lambda _joints, _mode, _step: (-3,)

        with pytest.raises(driver.JakaError, match=r"servo_j failed \(-3\)"):
            arm.servo_enable(True)
    finally:
        arm.disconnect()


def test_cartesian_servo_starts_with_servo_p_not_servo_j(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2", auto_enable_servo=False))
    arm.connect()
    try:
        arm.servo_enable(True, representation="eef")

        wait_until(lambda: bool(servo_calls(rc, "servo_p")))
        assert not servo_calls(rc, "servo_j")
        assert servo_calls(rc, "servo_p")[0][1] == pytest.approx(rc.tcp_mm)
    finally:
        arm.disconnect()


def test_servo_target_timeout_disables_controller(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2", servo_target_timeout_s=0.025))
    arm.connect()
    try:
        wait_until(lambda: not arm.get_servo_status()["worker_alive"])
        status = arm.get_servo_status()
        assert status["active"] is False
        assert "target timeout" in status["watchdog"]
        assert ("servo_move_enable", False) in rc.calls
    finally:
        arm.disconnect()


def test_servo_sdk_failure_stops_sender_and_records_error(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(driver, "create_rc", lambda _ip: rc)
    arm = driver.JakaRobot(driver.JakaRobotConfig(ip="10.0.0.2"))
    arm.connect()
    try:
        rc.servo_j = lambda _joints, _mode, _step: (-3,)
        wait_until(lambda: not arm.get_servo_status()["worker_alive"])
        status = arm.get_servo_status()
        assert status["active"] is False
        assert "servo_j failed (-3)" in status["last_error"]
        assert ("servo_move_enable", False) in rc.calls
        with pytest.raises(RuntimeError, match=r"Last worker error: JAKA servo_j failed \(-3\)"):
            arm.send_action({"joint_1.pos": 0.1})
    finally:
        arm.disconnect()
