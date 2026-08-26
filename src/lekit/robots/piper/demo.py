"""Rich-powered static and dynamic self-test for a physical Piper arm."""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from .piper_robot import PiperRobot, PiperRobotConfig

JOINT_STEP_RAD = 0.01
GRIPPER_STEP_M = 0.005
DEMO_SPEED_PERCENT = 10
MOTION_TIMEOUT_S = 4.0
COMMUNICATION_WARMUP_S = 1.0
FEEDBACK_WARMUP_S = 1.0
JOINT_TOLERANCE_RAD = 0.003
GRIPPER_TOLERANCE_M = 0.0015

JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
DRIVER_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "stall_status",
)
GRIPPER_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
)
DRIVER_STATUS_FIELDS = (*DRIVER_FAULT_FIELDS, "driver_enable_status")
GRIPPER_STATUS_FIELDS = (*GRIPPER_FAULT_FIELDS, "driver_enable_status", "homing_status")


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


STATUS_STYLES = {
    CheckStatus.PASS: "bold green",
    CheckStatus.WARN: "bold yellow",
    CheckStatus.FAIL: "bold red",
    CheckStatus.SKIP: "dim cyan",
}


@dataclass(frozen=True)
class CheckResult:
    category: str
    name: str
    status: CheckStatus
    detail: str
    duration_s: float = 0.0
    blocks_motion: bool = False


class PiperSelfTest:
    """Collect Piper diagnostics and render one consolidated report."""

    def __init__(self, robot: PiperRobot, console: Console):
        self.robot = robot
        self.console = console
        self.results: list[CheckResult] = []
        self.initial_observation: dict[str, float] = {}
        self.gripper_homed = False
        self.joint_motion_active = False
        self.gripper_motion_active = False

    @property
    def has_failures(self) -> bool:
        return any(result.status is CheckStatus.FAIL for result in self.results)

    @property
    def motion_blocked(self) -> bool:
        return any(result.status is CheckStatus.FAIL and result.blocks_motion for result in self.results)

    def add(
        self,
        category: str,
        name: str,
        status: CheckStatus,
        detail: str,
        *,
        started_at: float | None = None,
        blocks_motion: bool = False,
    ) -> None:
        duration = 0.0 if started_at is None else time.monotonic() - started_at
        self.results.append(CheckResult(category, name, status, detail, duration, blocks_motion))

    def run_static_checks(self) -> None:
        self.console.rule("[bold cyan]静态检查[/bold cyan]")
        self._check_communication_and_firmware()
        self._check_arm_state()
        self._check_joint_observation()
        self._check_flange_pose()
        self._check_motors_and_drivers()
        self._check_gripper()

    def _check_communication_and_firmware(self) -> None:
        arm = self._arm()
        started_at = time.monotonic()
        try:
            deadline = time.monotonic() + COMMUNICATION_WARMUP_S
            while True:
                has_error = bool(arm.has_comm_error())
                error = arm.get_comm_error()
                is_ok = bool(arm.is_ok())
                fps = float(arm.get_fps())
                if has_error or error is not None or (is_ok and math.isfinite(fps) and fps > 0):
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            if has_error or error is not None or not is_ok or not math.isfinite(fps) or fps <= 0:
                detail = f"is_ok={is_ok}, fps={fps:.1f} Hz, error={error!s}"
                self.add(
                    "通信与固件",
                    "CAN / SDK 通信",
                    CheckStatus.FAIL,
                    detail,
                    started_at=started_at,
                    blocks_motion=True,
                )
            else:
                self.add(
                    "通信与固件",
                    "CAN / SDK 通信",
                    CheckStatus.PASS,
                    f"通信正常，接收频率 {fps:.1f} Hz",
                    started_at=started_at,
                )
        except Exception as exc:
            self.add(
                "通信与固件",
                "CAN / SDK 通信",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )

        started_at = time.monotonic()
        try:
            firmware = arm.get_firmware()
            if not isinstance(firmware, dict) or not firmware.get("software_version"):
                self.add(
                    "通信与固件",
                    "固件信息",
                    CheckStatus.FAIL,
                    "未收到可识别的软件固件版本，无法确定兼容驱动",
                    started_at=started_at,
                    blocks_motion=True,
                )
                return
            detail = ", ".join(
                f"{key}={firmware[key]}"
                for key in ("software_version", "hardware_version", "production_date")
                if firmware.get(key) is not None
            )
            expected_driver = self._expected_firmware_driver(str(firmware["software_version"]))
            configured_driver = self.robot.config.firmware_version
            if expected_driver is None:
                self.add(
                    "通信与固件",
                    "固件信息",
                    CheckStatus.FAIL,
                    f"{detail}; 无法确定兼容驱动",
                    started_at=started_at,
                    blocks_motion=True,
                )
            elif configured_driver != expected_driver:
                self.add(
                    "通信与固件",
                    "固件信息",
                    CheckStatus.FAIL,
                    f"{detail}; 当前驱动={configured_driver}，请使用 --firmware-version {expected_driver}",
                    started_at=started_at,
                    blocks_motion=True,
                )
            else:
                self.add("通信与固件", "固件信息", CheckStatus.PASS, detail, started_at=started_at)
        except Exception as exc:
            self.add(
                "通信与固件",
                "固件信息",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )

    @staticmethod
    def _expected_firmware_driver(software_version: str) -> str | None:
        match = re.fullmatch(r"S-V1\.8-(\d+)", software_version)
        if match is None:
            return None
        patch = int(match.group(1))
        if patch >= 9:
            return "v189"
        if patch == 8:
            return "v188"
        if patch >= 3:
            return "v183"
        return "default"

    def _check_arm_state(self) -> None:
        arm = self._arm()
        started_at = time.monotonic()
        try:
            status = arm.get_arm_status()
            message = getattr(status, "msg", None)
            if message is None:
                raise RuntimeError("未收到整机状态反馈")
            arm_status = int(message.arm_status)
            err_code = int(message.err_code)
            if arm_status != 0 or err_code != 0:
                self.add(
                    "机械臂状态",
                    "控制器状态",
                    CheckStatus.FAIL,
                    f"arm_status={arm_status}, err_code=0x{err_code:04X}",
                    started_at=started_at,
                    blocks_motion=True,
                )
            else:
                self.add(
                    "机械臂状态",
                    "控制器状态",
                    CheckStatus.PASS,
                    f"状态正常，ctrl_mode={message.ctrl_mode}, motion_status={message.motion_status}",
                    started_at=started_at,
                )
        except Exception as exc:
            self.add(
                "机械臂状态",
                "控制器状态",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )

        started_at = time.monotonic()
        try:
            enabled = bool(arm.get_joint_limits_enabled())
            status = CheckStatus.PASS if enabled else CheckStatus.FAIL
            self.add(
                "机械臂状态",
                "SDK 软件关节限位",
                status,
                "已启用" if enabled else "未启用",
                started_at=started_at,
                blocks_motion=not enabled,
            )
        except Exception as exc:
            self.add(
                "机械臂状态",
                "SDK 软件关节限位",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )

    def _check_joint_observation(self) -> None:
        started_at = time.monotonic()
        try:
            observation = self.robot.get_observation()
            joints = {key: float(observation[key]) for key in JOINT_KEYS}
            if any(not math.isfinite(value) for value in joints.values()):
                raise RuntimeError("关节反馈包含非有限值")
            self._validate_joint_limits(joints)
            self.initial_observation = {
                key: float(value)
                for key, value in observation.items()
                if key in JOINT_KEYS or key == "gripper.pos"
            }
            detail = ", ".join(f"J{index}={joints[key]:+.4f}" for index, key in enumerate(JOINT_KEYS, 1))
            self.add(
                "机械臂状态",
                "LeRobot 关节反馈",
                CheckStatus.PASS,
                detail + " rad",
                started_at=started_at,
            )
        except Exception as exc:
            self.add(
                "机械臂状态",
                "LeRobot 关节反馈",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )

    def _validate_joint_limits(self, joints: dict[str, float]) -> None:
        limits = getattr(self.robot, "_joint_limits", None)
        if not isinstance(limits, dict):
            return
        for key, value in joints.items():
            lower, upper = limits[key]
            if value < lower or value > upper:
                raise RuntimeError(
                    f"{key}={value:.6f} rad 超出模型限位 [{lower:.6f}, {upper:.6f}]，无法安全原位往返"
                )

    def _check_flange_pose(self) -> None:
        started_at = time.monotonic()
        try:
            feedback = self._arm().get_flange_pose()
            pose = [float(value) for value in getattr(feedback, "msg", [])]
            if len(pose) != 6 or any(not math.isfinite(value) for value in pose):
                raise RuntimeError("法兰位姿反馈缺失或格式错误")
            detail = f"xyz=({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}) m"
            self.add("机械臂状态", "法兰位姿", CheckStatus.PASS, detail, started_at=started_at)
        except Exception as exc:
            self.add(
                "机械臂状态",
                "法兰位姿",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )

    def _check_motors_and_drivers(self) -> None:
        arm = self._arm()
        for joint_index in range(1, 7):
            started_at = time.monotonic()
            try:
                deadline = time.monotonic() + FEEDBACK_WARMUP_S
                while True:
                    motor = arm.get_motor_states(joint_index)
                    driver = arm.get_driver_states(joint_index)
                    motor_msg = getattr(motor, "msg", None)
                    driver_msg = getattr(driver, "msg", None)
                    motor_hz = float(getattr(motor, "hz", 0.0))
                    driver_hz = float(getattr(driver, "hz", 0.0))
                    ready = (
                        motor_msg is not None
                        and driver_msg is not None
                        and math.isfinite(motor_hz)
                        and math.isfinite(driver_hz)
                        and motor_hz > 0
                        and driver_hz > 0
                    )
                    if ready or time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                if motor_msg is None or driver_msg is None:
                    raise RuntimeError("电机或驱动器反馈缺失")
                if (
                    not math.isfinite(motor_hz)
                    or not math.isfinite(driver_hz)
                    or motor_hz <= 0
                    or driver_hz <= 0
                ):
                    raise RuntimeError(f"反馈频率异常: motor={motor_hz:.1f} Hz, driver={driver_hz:.1f} Hz")
                motor_values = (
                    float(motor_msg.position),
                    float(motor_msg.velocity),
                    float(motor_msg.current),
                    float(motor_msg.torque),
                )
                driver_values = (
                    float(driver_msg.vol),
                    float(driver_msg.foc_temp),
                    float(driver_msg.motor_temp),
                    float(driver_msg.bus_current),
                )
                if any(not math.isfinite(value) for value in (*motor_values, *driver_values)):
                    raise RuntimeError("电机或驱动器反馈包含非有限值")
                faults = self._active_faults(
                    driver_msg.foc_status,
                    DRIVER_FAULT_FIELDS,
                    required_fields=DRIVER_STATUS_FIELDS,
                )
                if faults:
                    self.add(
                        "六轴电机与驱动器",
                        f"关节 {joint_index}",
                        CheckStatus.FAIL,
                        "活动故障位: " + ", ".join(faults),
                        started_at=started_at,
                        blocks_motion=True,
                    )
                    continue
                self.add(
                    "六轴电机与驱动器",
                    f"关节 {joint_index}",
                    CheckStatus.PASS,
                    f"{driver_values[0]:.1f} V, 电机 {driver_values[2]:.1f} °C, "
                    f"电流 {motor_values[2]:.2f} A, 扭矩 {motor_values[3]:.3f} N·m",
                    started_at=started_at,
                )
            except Exception as exc:
                self.add(
                    "六轴电机与驱动器",
                    f"关节 {joint_index}",
                    CheckStatus.FAIL,
                    str(exc),
                    started_at=started_at,
                    blocks_motion=True,
                )

    def _check_gripper(self) -> None:
        if self.robot.gripper is None:
            self.add("夹爪", "AGX 夹爪", CheckStatus.SKIP, "配置为不使用夹爪")
            return
        started_at = time.monotonic()
        try:
            gripper = self.robot.gripper
            deadline = time.monotonic() + FEEDBACK_WARMUP_S
            while True:
                is_ok = bool(gripper.is_ok())
                fps = float(gripper.get_fps())
                feedback = gripper.get_gripper_status()
                message = getattr(feedback, "msg", None)
                ready = is_ok and math.isfinite(fps) and fps > 0 and message is not None
                if ready or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            if not is_ok or not math.isfinite(fps) or fps <= 0 or message is None:
                raise RuntimeError(f"夹爪通信异常: is_ok={is_ok}, fps={fps:.1f} Hz")
            width = float(message.value)
            force = float(message.force)
            if message.mode != "width" or not math.isfinite(width) or not math.isfinite(force):
                raise RuntimeError("夹爪反馈格式错误")
            if not self.robot.config.gripper_min_width_m <= width <= self.robot.config.gripper_max_width_m:
                raise RuntimeError(f"夹爪宽度 {width:.6f} m 超出配置范围")
            faults = self._active_faults(
                message.foc_status,
                GRIPPER_FAULT_FIELDS,
                required_fields=GRIPPER_STATUS_FIELDS,
            )
            if faults:
                self.add(
                    "夹爪",
                    "AGX 夹爪",
                    CheckStatus.FAIL,
                    "活动故障位: " + ", ".join(faults),
                    started_at=started_at,
                    blocks_motion=True,
                )
                return
            self.gripper_homed = bool(getattr(message.foc_status, "homing_status", False))
            status = CheckStatus.PASS if self.gripper_homed else CheckStatus.WARN
            homing = "已归零" if self.gripper_homed else "未归零，动态夹爪检查将跳过"
            self.add(
                "夹爪",
                "AGX 夹爪",
                status,
                f"{width:.4f} m, {force:.2f} N, {fps:.1f} Hz, {homing}",
                started_at=started_at,
            )
        except Exception as exc:
            self.add(
                "夹爪",
                "AGX 夹爪",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )

    @staticmethod
    def _active_faults(
        status: Any,
        fields: tuple[str, ...],
        *,
        required_fields: tuple[str, ...],
    ) -> list[str]:
        missing = [field for field in required_fields if status is None or not hasattr(status, field)]
        if missing:
            raise RuntimeError("故障状态反馈不完整，缺少: " + ", ".join(missing))
        return [field for field in fields if bool(getattr(status, field, False))]

    def run_dynamic_checks(self, confirm_motion: Callable[[], bool]) -> None:
        self.console.rule("[bold magenta]动态运动检查[/bold magenta]")
        if self.motion_blocked:
            self.add(
                "动态运动检查",
                "安全门",
                CheckStatus.SKIP,
                "静态安全检查失败，禁止使能和运动",
            )
            return
        if not confirm_motion():
            self.add("动态运动检查", "用户确认", CheckStatus.SKIP, "用户取消，未使能且未发送动作")
            return
        try:
            if not self._revalidate_motion_safety():
                return
            if not self._enable_arm():
                return
            if not self._check_joint_motion():
                return
            self._check_gripper_motion()
        except KeyboardInterrupt:
            self.add(
                "动态运动检查",
                "用户中断",
                CheckStatus.FAIL,
                "收到 Ctrl+C，已中止后续检查并尝试恢复正在测试的机构",
                blocks_motion=True,
            )
            self.console.print("[bold red]收到用户中断，正在尝试恢复安全起点…[/bold red]")
            if self.joint_motion_active:
                self._attempt_full_joint_restore()
            if self.gripper_motion_active:
                self._attempt_gripper_restore("夹爪中断恢复")

    def _revalidate_motion_safety(self) -> bool:
        started_at = time.monotonic()
        try:
            arm = self._arm()
            communication_error = arm.get_comm_error()
            if arm.has_comm_error() or communication_error is not None or not arm.is_ok():
                raise RuntimeError(f"通信状态异常: {communication_error!s}")
            status = arm.get_arm_status()
            status_message = getattr(status, "msg", None)
            if (
                status_message is None
                or int(status_message.arm_status) != 0
                or int(status_message.err_code) != 0
            ):
                raise RuntimeError("控制器状态在确认期间发生变化")
            if not arm.get_joint_limits_enabled():
                raise RuntimeError("SDK 软件关节限位已关闭")
            observation = self.robot.get_observation()
            joints = {key: float(observation[key]) for key in JOINT_KEYS}
            if any(not math.isfinite(value) for value in joints.values()):
                raise RuntimeError("关节反馈包含非有限值")
            self._validate_joint_limits(joints)
            for joint_index in range(1, 7):
                driver = arm.get_driver_states(joint_index)
                driver_message = getattr(driver, "msg", None)
                if driver_message is None:
                    raise RuntimeError(f"关节 {joint_index} 驱动器反馈缺失")
                faults = self._active_faults(
                    driver_message.foc_status,
                    DRIVER_FAULT_FIELDS,
                    required_fields=DRIVER_STATUS_FIELDS,
                )
                if faults:
                    raise RuntimeError(f"关节 {joint_index} 活动故障位: {', '.join(faults)}")
            if self.robot.gripper is not None:
                gripper_feedback = self.robot.gripper.get_gripper_status()
                gripper_message = getattr(gripper_feedback, "msg", None)
                if gripper_message is None:
                    raise RuntimeError("夹爪反馈缺失")
                gripper_faults = self._active_faults(
                    gripper_message.foc_status,
                    GRIPPER_FAULT_FIELDS,
                    required_fields=GRIPPER_STATUS_FIELDS,
                )
                if gripper_faults:
                    raise RuntimeError("夹爪活动故障位: " + ", ".join(gripper_faults))
                self.gripper_homed = bool(gripper_message.foc_status.homing_status)
            self.initial_observation = {
                key: float(value)
                for key, value in observation.items()
                if key in JOINT_KEYS or key == "gripper.pos"
            }
            self.add(
                "动态运动检查",
                "确认后安全复核",
                CheckStatus.PASS,
                "通信、控制器、限位和六轴驱动器正常；已刷新运动基线",
                started_at=started_at,
            )
            return True
        except Exception as exc:
            self.add(
                "动态运动检查",
                "确认后安全复核",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )
            return False

    def _enable_arm(self) -> bool:
        started_at = time.monotonic()
        arm = self._arm()
        try:
            deadline = time.monotonic() + MOTION_TIMEOUT_S
            while not arm.enable():
                if time.monotonic() >= deadline:
                    raise TimeoutError("机械臂使能超时")
                time.sleep(0.05)
            enabled = list(arm.get_joints_enable_status_list())
            if len(enabled) != 6 or not all(enabled):
                raise RuntimeError(f"关节使能状态异常: {enabled}")
            self.add(
                "动态运动检查",
                "六轴使能",
                CheckStatus.PASS,
                "六个关节均已使能",
                started_at=started_at,
            )
            return True
        except Exception as exc:
            self.add(
                "动态运动检查",
                "六轴使能",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
                blocks_motion=True,
            )
            return False

    def _check_joint_motion(self) -> bool:
        if not self.initial_observation:
            self.add("动态运动检查", "六轴往返", CheckStatus.FAIL, "缺少初始关节反馈")
            return False
        for joint_index, key in enumerate(JOINT_KEYS, 1):
            started_at = time.monotonic()
            start = self.initial_observation[key]
            target = self._safe_joint_probe_target(key, start)
            try:
                self.joint_motion_active = True
                applied = self.robot.send_action({key: target})
                reached = self._wait_for_value(key, float(applied[key]), JOINT_TOLERANCE_RAD)
                self.robot.send_action({key: start})
                restored = self._wait_for_value(key, start, JOINT_TOLERANCE_RAD)
                self.joint_motion_active = False
                self.add(
                    "动态运动检查",
                    f"关节 {joint_index} 往返",
                    CheckStatus.PASS,
                    f"起点 {start:+.5f} → 探测 {reached:+.5f} → 返回 {restored:+.5f} rad",
                    started_at=started_at,
                )
            except Exception as exc:
                self.add(
                    "动态运动检查",
                    f"关节 {joint_index} 往返",
                    CheckStatus.FAIL,
                    str(exc),
                    started_at=started_at,
                    blocks_motion=True,
                )
                self.console.print("[bold red]中止剩余动态检查，正在尝试恢复完整初始位姿…[/bold red]")
                self._attempt_full_joint_restore()
                return False
        return True

    def _safe_joint_probe_target(self, key: str, start: float) -> float:
        positive = start + JOINT_STEP_RAD
        limits = getattr(self.robot, "_joint_limits", None)
        if not isinstance(limits, dict):
            return positive
        lower, upper = limits[key]
        margin = JOINT_TOLERANCE_RAD
        if positive <= upper - margin:
            return positive
        negative = start - JOINT_STEP_RAD
        if negative >= lower + margin:
            return negative
        raise RuntimeError(f"{key} 距模型限位过近，无法安全执行 {JOINT_STEP_RAD:g} rad 探测")

    def _attempt_full_joint_restore(self) -> None:
        started_at = time.monotonic()
        action = {key: self.initial_observation[key] for key in JOINT_KEYS}
        try:
            self.robot.send_action(action)
            for key, target in action.items():
                self._wait_for_value(key, target, JOINT_TOLERANCE_RAD)
            self.add(
                "动态运动检查",
                "故障恢复",
                CheckStatus.PASS,
                "已恢复完整初始关节位姿",
                started_at=started_at,
            )
            self.joint_motion_active = False
        except Exception as exc:
            self.add(
                "动态运动检查",
                "故障恢复",
                CheckStatus.FAIL,
                f"无法确认已恢复初始位姿: {exc}",
                started_at=started_at,
                blocks_motion=True,
            )

    def _check_gripper_motion(self) -> None:
        if self.robot.gripper is None:
            self.add("动态运动检查", "夹爪往返", CheckStatus.SKIP, "未配置夹爪")
            return
        if not self.gripper_homed:
            self.add("动态运动检查", "夹爪往返", CheckStatus.SKIP, "夹爪未归零，禁止动作")
            return
        started_at = time.monotonic()
        start = self.initial_observation.get("gripper.pos")
        try:
            if start is None:
                raise RuntimeError("运动基线缺少 gripper.pos")
            target = self._safe_gripper_probe_target(start)
            self.gripper_motion_active = True
            self.robot.send_action({"gripper.pos": target})
            reached = self._wait_for_value("gripper.pos", target, GRIPPER_TOLERANCE_M)
            self.robot.send_action({"gripper.pos": start})
            restored = self._wait_for_value("gripper.pos", start, GRIPPER_TOLERANCE_M)
            self.gripper_motion_active = False
            self.add(
                "动态运动检查",
                "夹爪往返",
                CheckStatus.PASS,
                f"起点 {start:.4f} → 探测 {reached:.4f} → 返回 {restored:.4f} m",
                started_at=started_at,
            )
        except Exception as exc:
            self.add(
                "动态运动检查",
                "夹爪往返",
                CheckStatus.FAIL,
                str(exc),
                started_at=started_at,
            )
            if start is not None and self.gripper_motion_active:
                self._attempt_gripper_restore("夹爪故障恢复")

    def _attempt_gripper_restore(self, name: str) -> None:
        started_at = time.monotonic()
        try:
            start = self.initial_observation["gripper.pos"]
            self.robot.send_action({"gripper.pos": start})
            self._wait_for_value("gripper.pos", start, GRIPPER_TOLERANCE_M)
            self.gripper_motion_active = False
            self.add(
                "动态运动检查",
                name,
                CheckStatus.PASS,
                "已恢复夹爪初始宽度",
                started_at=started_at,
            )
        except Exception as exc:
            self.add(
                "动态运动检查",
                name,
                CheckStatus.FAIL,
                f"无法确认夹爪已恢复: {exc}",
                started_at=started_at,
            )

    def _safe_gripper_probe_target(self, start: float) -> float:
        maximum = self.robot.config.gripper_max_width_m
        minimum = self.robot.config.gripper_min_width_m
        if start + GRIPPER_STEP_M <= maximum:
            return start + GRIPPER_STEP_M
        if start - GRIPPER_STEP_M >= minimum:
            return start - GRIPPER_STEP_M
        raise RuntimeError("夹爪当前位置没有足够空间执行 5 mm 探测")

    def _wait_for_value(self, key: str, target: float, tolerance: float) -> float:
        deadline = time.monotonic() + MOTION_TIMEOUT_S
        while True:
            measured = float(self.robot.get_observation()[key])
            if math.isfinite(measured) and abs(measured - target) <= tolerance:
                return measured
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{key} 在 {MOTION_TIMEOUT_S:g}s 内未到达目标: target={target:.6f}, measured={measured:.6f}"
                )
            time.sleep(0.05)

    def _arm(self) -> Any:
        if self.robot.arm is None:
            raise RuntimeError("Piper SDK 驱动未初始化")
        return self.robot.arm

    def render_report(self) -> None:
        categories = dict.fromkeys(result.category for result in self.results)
        for category in categories:
            table = Table(
                title=category,
                box=box.ROUNDED,
                header_style="bold cyan",
                expand=True,
                show_lines=False,
            )
            table.add_column("状态", width=8, justify="center", no_wrap=True)
            table.add_column("检查项", width=24, style="bold")
            table.add_column("结果")
            table.add_column("耗时", width=9, justify="right")
            for result in (item for item in self.results if item.category == category):
                status = Text(result.status.value, style=STATUS_STYLES[result.status])
                duration = "—" if result.duration_s <= 0 else f"{result.duration_s * 1000:.0f} ms"
                table.add_row(status, result.name, result.detail, duration)
            self.console.print(table)

        counts = Counter(result.status for result in self.results)
        summary = (
            f"[green]PASS {counts[CheckStatus.PASS]}[/green]  "
            f"[yellow]WARN {counts[CheckStatus.WARN]}[/yellow]  "
            f"[red]FAIL {counts[CheckStatus.FAIL]}[/red]  "
            f"[cyan]SKIP {counts[CheckStatus.SKIP]}[/cyan]"
        )
        if self.has_failures:
            conclusion = "[bold red]发现故障，请根据报告处理后重新自检。[/bold red]"
            border_style = "red"
        else:
            conclusion = "[bold green]全部关键检查通过。[/bold green]"
            border_style = "green"
        self.console.print(
            Panel.fit(
                f"{summary}\n{conclusion}",
                title="[bold]自检结论[/bold]",
                border_style=border_style,
                padding=(1, 3),
            )
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a formatted Piper arm self-test.")
    parser.add_argument("--channel", default="can0", help="CAN channel (default: can0)")
    parser.add_argument(
        "--firmware-version",
        choices=("default", "v183", "v188", "v189"),
        default="default",
    )
    parser.add_argument(
        "--robot-model",
        choices=("piper", "piper_h", "piper_l", "piper_x"),
        default="piper",
    )
    parser.add_argument("--no-gripper", action="store_true", help="Skip AGX gripper checks")
    parser.add_argument(
        "--full",
        action="store_true",
        help="After static checks and confirmation, test all joints and the gripper with small return motions",
    )
    return parser


def _default_confirmation(console: Console) -> bool:
    console.print(
        Panel(
            "即将以 10% 速度依次测试六个关节（每轴 0.01 rad）和夹爪（5 mm），并返回初始位置。\n"
            "请清空机械臂工作空间、确认急停可用，并持续观察机械臂。",
            title="[bold yellow]动态自检安全确认[/bold yellow]",
            border_style="yellow",
        )
    )
    return Confirm.ask("确认开始动态自检？", console=console, default=False)


def main(
    argv: Sequence[str] | None = None,
    *,
    robot_factory: Callable[[PiperRobotConfig], PiperRobot] = PiperRobot,
    console: Console | None = None,
    confirm_motion: Callable[[], bool] | None = None,
) -> int:
    """Run static diagnostics and optional operator-confirmed motion checks."""

    args = _build_parser().parse_args(argv)
    console = console or Console()
    console.print(
        Panel.fit(
            f"[bold]型号[/bold]  {args.robot_model}\n"
            f"[bold]通道[/bold]  {args.channel}\n"
            f"[bold]模式[/bold]  {'完整动态自检' if args.full else '静态安全检查'}",
            title="[bold cyan]Piper 完全自检[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    config = PiperRobotConfig(
        channel=args.channel,
        firmware_version=args.firmware_version,
        robot_model=args.robot_model,
        include_gripper=not args.no_gripper,
        auto_enable=False,
        speed_percent=DEMO_SPEED_PERCENT,
        max_relative_target=0.05,
    )
    robot = robot_factory(config)
    self_test = PiperSelfTest(robot, console)

    try:
        with robot:
            self_test.add("通信与固件", "LeRobot 连接", CheckStatus.PASS, f"已连接 {args.channel}")
            self_test.run_static_checks()
            if args.full:
                confirmation = confirm_motion or (lambda: _default_confirmation(console))
                self_test.run_dynamic_checks(confirmation)
            else:
                self_test.add(
                    "动态运动检查",
                    "运行模式",
                    CheckStatus.SKIP,
                    "动态自检未启用；使用 --full 并人工确认后执行",
                )
    except KeyboardInterrupt:
        self_test.add(
            "动态运动检查" if args.full else "通信与固件",
            "用户中断",
            CheckStatus.FAIL,
            "收到 Ctrl+C，自检已终止",
            blocks_motion=True,
        )
    except Exception as exc:
        self_test.add(
            "通信与固件",
            "LeRobot 连接 / 自检执行",
            CheckStatus.FAIL,
            f"{type(exc).__name__}: {exc}",
            blocks_motion=True,
        )

    self_test.render_report()
    return 1 if self_test.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
