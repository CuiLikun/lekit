import time
from dataclasses import dataclass, field
from functools import cached_property
from platform import system
from typing import Any, ClassVar

import numpy as np

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

try:
    import pyAgxArm as pyagxarm  # type: ignore[import-not-found]  # noqa: N813
except ImportError as e:
    raise ImportError("pyAgxArm is not installed") from e


def resolve_interface_and_channel() -> tuple[str, str]:
    """Resolve the default CAN interface and channel for the current platform."""
    sys_map = {
        "Windows": ("agx_cando", "0"),
        "Linux": ("socketcan", "can0"),
        "Darwin": ("slcan", "/dev/ttyACM0"),
    }
    assert system() in sys_map, f"Unsupported platform: {system()}"
    return sys_map[system()]


@RobotConfig.register_subclass("agx_arm")
@dataclass
class AgxArmConfig(RobotConfig):
    model: str = "piper_x"
    firmware: str = "v188"
    # CAN interface / channel. ``socketcan`` + ``can0`` matches a Linux
    # SocketCAN setup; Windows / macOS users typically swap to ``agx_cando``
    # / ``slcan`` + the matching USB channel.
    interface: str = resolve_interface_and_channel()[0]
    channel: str = resolve_interface_and_channel()[1]
    bitrate: int = 1_000_000
    # Effector kind (only ``agx_gripper`` is wired up here).
    effector: str = "agx_gripper"
    # Whether ``connect()`` also powers-on and enables the servos so the
    # first ``send_action`` is immediately ready.
    auto_enable: bool = True
    # Best-effort clear of any joint-error state on connect (matches
    # demo.py: ``robot.clear_joint_error()``).
    auto_clear_joint_error: bool = True
    # Engage the firmware's leader/follower linkage mode (follower side).
    # Demo.py calls ``robot.set_follower_mode()`` before driving commands;
    # we keep the same default.
    auto_set_follower_mode: bool = False
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class AgxArm(Robot):
    config_class: ClassVar[type] = AgxArmConfig
    name: ClassVar[str] = "agx_arm"
    motors: list[str] = []

    def __init__(self, config: AgxArmConfig):
        super().__init__(config)
        self.config: AgxArmConfig = config
        self.arm: Any | None = None
        self.eef: Any | None = None
        self.status: dict[str, Any] = {}
        self.cameras: dict[str, Any] = make_cameras_from_configs(config.cameras) if config.cameras else {}

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {f"{motor}.pos": float for motor in self.motors}
        features["gripper.pos"] = float
        features.update({f"ee.{axis}": float for axis in ("x", "y", "z", "roll", "pitch", "yaw")})
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_key] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{cam_key}_depth"] = (cam.height, cam.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {f"{motor}.pos": float for motor in self.motors}
        features["gripper.pos"] = float
        features.update({f"ee.{axis}": float for axis in ("x", "y", "z", "roll", "pitch", "yaw")})
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self.arm is not None
            and self.arm.is_connected()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # The AgileX controller handles joint zeroing internally; the host
        # has no per-motor calibration step (matches the demo workflow).
        return True

    def calibrate(self) -> None:
        return

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Open the CAN bus, configure the arm, and connect cameras."""
        sdk_config = pyagxarm.create_agx_arm_config(
            robot=self.config.model,
            firmware_version=self.config.firmware,
            interface=self.config.interface,
            channel=self.config.channel,
            bitrate=self.config.bitrate,
        )
        self.arm = pyagxarm.AgxArmFactory.create_arm(sdk_config)
        try:
            self.arm.connect()
            if self.arm.has_comm_error():
                print(f"Detected communication error: {self.arm.get_comm_error()}")

            if self.config.auto_clear_joint_error and self.arm.clear_joint_error() is False:
                print("AgxArm: controller did not acknowledge clear_joint_error().")

            if self.config.auto_enable:
                self.arm.enable()
                self.arm.set_speed_percent(100)

            if self.config.auto_set_follower_mode:
                self.arm.set_follower_mode()

            self.eef = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

            for camera in self.cameras.values():
                camera.connect()
            self.configure()
        except Exception:
            print("AgxArm: failed to connect; cleaning up.")
            self.disconnect()
            raise
        self.motors = [f"joint_{i + 1}" for i in range(self.arm.joint_nums)]
        print(f"AgxArm[{self.arm.joint_nums}DOF] connected.")
        print(f"firmware={self.arm.get_firmware()}")
        print(f"status={self.arm.is_ok()}")
        print(f"fps={self.arm.get_fps()}")
        print(f"arm_status={self.arm.get_arm_status()}")
        print(f"model={self.config.model}")
        print(f"id={self.id}")
        print(f"firmware={self.config.firmware}")
        print(f"interface={self.config.interface}")
        print(f"channel={self.config.channel}")
        print(f"joint_enable_status_list={self.arm.get_joints_enable_status_list()}")

    @check_if_not_connected
    def check_status(self) -> dict[str, Any]:
        """Collect the latest SDK health signals and store them in ``self.status``.

        ``pyAgxArm`` exposes communication health separately from the arm's
        controller feedback. Keep both so callers can distinguish a CAN
        transport fault from a disabled joint or an arm-side error state.
        """
        arm = self.arm
        comm_error = bool(arm.has_comm_error())
        joint_enable_status = [bool(enabled) for enabled in arm.get_joints_enable_status_list()]
        arm_status = arm.get_arm_status()
        arm_status_message = None if arm_status is None else arm_status.msg
        gripper_ok = False
        if self.eef is not None:
            try:
                gripper_ok = self.eef.get_gripper_status() is not None
            except Exception:
                # A missing gripper feedback should be reported in the panel,
                # while a transient SDK read failure must not hide arm status.
                gripper_ok = False
        self.status = {
            "connected": bool(arm.is_connected()),
            "is_ok": bool(arm.is_ok()),
            "comm_error": comm_error,
            "comm_error_detail": arm.get_comm_error() if comm_error else None,
            "fps": float(arm.get_fps()),
            "joint_enable_status": joint_enable_status,
            "all_joints_enabled": bool(joint_enable_status) and all(joint_enable_status),
            "arm_status": arm_status_message,
            "arm_status_hz": None if arm_status is None else arm_status.hz,
            "arm_status_timestamp": None if arm_status is None else arm_status.timestamp,
            "gripper_ok": gripper_ok,
        }
        return self.status

    def configure(self) -> None:
        """Enable the SDK's model-specific software joint limits."""
        if self.arm is not None:
            self.arm.set_joint_limits_enabled(True)

    def disconnect(self) -> None:
        """Best-effort teardown of torque, CAN resources, and cameras."""
        if self.arm is not None:
            try:
                self.arm.disconnect()
            except Exception as e:
                print("AgxArm: error during disconnect: %s", e)
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception:
                print("AgxArm: camera disconnect failed, ignoring.")
        self.arm = None
        self.eef = None
        print(f"AgxArm[{self.arm.joint_nums}DOF] disconnected.")

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        """Read arm state and configured camera frames."""
        obs: dict[str, Any] = {}
        obs.update(
            {f"{motor}.pos": angle for motor, angle in zip(self.motors, self.get_joint_angles(), strict=True)}
        )
        obs.update({"gripper.pos": self.get_gripper_status()})
        obs.update(
            {
                f"ee.{axis}": value
                for axis, value in zip(
                    ["x", "y", "z", "roll", "pitch", "yaw"], self.get_flange_pose(), strict=True
                )
            }
        )

        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                obs[camera_name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                obs[f"{camera_name}_depth"] = camera.read_latest_depth()
        return obs

    @check_if_not_connected
    def get_joint_angles(self) -> list[float]:
        """Read the configured model's measured joint angles in radians."""
        sample = self.arm.get_joint_angles()
        assert sample is not None
        angles = np.asarray(sample.msg, dtype=float)
        assert angles.shape == (self.arm.joint_nums,) and np.all(np.isfinite(angles))
        return [float(angle) for angle in angles]

    @check_if_not_connected
    def get_gripper_status(self) -> float:
        """Read normalized gripper position in [0, 1] as ``gripper.pos`` field."""
        status = None if self.eef is None else self.eef.get_gripper_status()
        assert status is not None
        assert status.msg.mode == "width"
        value = float(status.msg.value)
        assert np.isfinite(value)
        return value

    @check_if_not_connected
    def get_flange_pose(self) -> list[float]:
        sample = self.arm.get_flange_pose()
        assert sample is not None, "AgxArm: get_flange_pose() returned None."
        pose = np.asarray(sample.msg, dtype=float)
        assert pose.shape == (6,) and np.all(np.isfinite(pose))
        return [float(value) for value in pose]

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the configured joint-space or Cartesian control command."""
        if "ee.x" in action:
            target = list(action.values())
            self.arm.move_p(target[:6])
            self.eef.move_gripper_m(target[6])
        else:
            pass
        return action


__all__ = [
    "AgxArm",
    "AgxArmConfig",
]


if __name__ == "__main__":
    from platform import system

    from rich import box, print
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    def make_live_panel(
        *,
        step: int,
        target_fps: float,
        actual_fps: float | None,
        states: list[float],
        action: list[float],
        status: dict[str, Any],
    ) -> Panel:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold cyan")
        summary.add_column()
        summary.add_row("Step", str(step))
        summary.add_row("Loop FPS", "--" if actual_fps is None else f"{actual_fps:.2f}")
        summary.add_row("Target FPS", f"{target_fps:.2f}")
        summary.add_row("SDK FPS", f"{status['fps']:.2f}")
        summary.add_row("Connected", "[green]yes[/green]" if status["connected"] else "[red]no[/red]")
        summary.add_row("Arm OK", "[green]yes[/green]" if status["is_ok"] else "[red]no[/red]")
        summary.add_row(
            "Joints Enabled",
            "[green]yes[/green]" if status["all_joints_enabled"] else "[red]no[/red]",
        )
        summary.add_row("Joint States", str(status["joint_enable_status"]))
        summary.add_row("Gripper OK", "[green]yes[/green]" if status["gripper_ok"] else "[red]no[/red]")
        if status["comm_error"]:
            summary.add_row("Comm Error", f"[red]{status['comm_error_detail']}[/red]")

        vectors = Table(box=box.SIMPLE_HEAVY, expand=True)
        vectors.add_column("Signal", style="bold cyan", no_wrap=True)
        vectors.add_column("Values", overflow="fold")
        vectors.add_row("States", ", ".join(f"{value:.4f}" for value in states))
        vectors.add_row("Action", ", ".join(f"{value:.4f}" for value in action))

        return Panel(
            Group(summary, vectors),
            title="AGX Arm Live Monitor",
            border_style="cyan",
        )

    config = AgxArmConfig(id="agx_demo")
    robot = AgxArm(config)
    fps = 1
    with robot:
        step = 0
        previous_frame_time = time.perf_counter()
        with Live(refresh_per_second=4, transient=False) as live:
            while True:
                frame_start = time.perf_counter()
                frame_interval = frame_start - previous_frame_time
                previous_frame_time = frame_start

                pos = robot.get_flange_pose()
                gripper_pos = robot.get_gripper_status()
                action: dict[str, Any] = {
                    "ee.x": pos[0],
                    "ee.y": pos[1],
                    "ee.z": pos[2] + 0.005,
                    "ee.roll": pos[3],
                    "ee.pitch": pos[4],
                    "ee.yaw": pos[5],
                    "gripper.pos": 0.005,
                }
                applied_action = robot.send_action(action)
                status = robot.check_status()
                live.update(
                    make_live_panel(
                        step=step,
                        target_fps=fps,
                        actual_fps=1.0 / frame_interval if step else None,
                        states=pos + [gripper_pos],
                        action=list(action.values()),
                        status=status,
                    )
                )
                step += 1
                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - frame_start)))
