"""
Run the ROS2 robot.

The following commands are executed to automatically activate ROS2 environment when activating the conda environment:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'source /home/nvidia/release/ros2_ws/install/setup.zsh' > $CONDA_PREFIX/etc/conda/activate.d/ros2_setup.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/ros2_setup.sh
```
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Type

import cv2
import draccus
import numpy as np
import rclpy  # type: ignore[import-untyped]
import torch

# ROS2 imports
from fuxi_interface.action import MoveToJoints  # type: ignore[import-untyped]
from fuxi_interface.msg import (  # type: ignore[import-untyped]
    ChassisMoving,
    GripperStatus,
    JointsWaypoint,
    LiftHeight,
    SwitchStatus,
)
from rclpy.action import ActionClient  # type: ignore[import-untyped]
from rclpy.node import Node  # type: ignore[import-untyped]
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # type: ignore[import-untyped]

# Rich imports
from rich import print
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

# ROS2 imports
from sensor_msgs.msg import JointState  # type: ignore[import-untyped]

# Lerobot imports
from lerobot.common.robot_devices.cameras.utils import (
    CameraConfig,
    IntelRealSenseCameraConfig,  # noqa
    OpenCVCameraConfig,
    make_cameras_from_configs,
)
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import RobotChassisMoveError, RobotDeviceNotConnectedError

# Distance from the lift zero position to the ground
LIFT_HEIGHT_OFFSET = 0.645
LIFT_DEFAULT_HEIGHT = 0.55

CHASSIS_DIRECTION_MAP = {
    "staying": 0,
    "forward": 1,
    "backward": 2,
    "left": 3,
    "right": 4,
}

CHASSIS_STATUS_MAP = {
    "idle": 0,
    "moving": 1,
    "success": 2,
    "failure": 3,
}

# Use adaptive QoS that matches the publisher's QoS settings
# This is similar to ros2 topic echo behavior
COMMON_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# Adaptive QoS for maximum compatibility
ADAPTIVE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

CHASSIS_COMMAND_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


@dataclass
class TopicSpec:
    key: str  # Unique identifier for the topic
    msg_type: Type
    topic_name: str
    topic_type: str  # "sub" or "pub"
    required: bool = False  # Whether this topic must receive a message before the node is considered ready


TOPICS = [
    TopicSpec("joint_states", JointState, "/proprio/joint_states", "sub", required=True),
    TopicSpec("teleop_action", JointState, "/teleop/joint_states", "sub", required=True),
    TopicSpec("left_gripper_status", GripperStatus, "/left_gripper_status", "sub", required=True),
    TopicSpec("right_gripper_status", GripperStatus, "/right_gripper_status", "sub", required=True),
    TopicSpec("switch_status", SwitchStatus, "/switch_status", "sub"),
    TopicSpec("lift_height", LiftHeight, "/lift_height", "sub"),
    TopicSpec("chassis_move_feedback", ChassisMoving, "/agent/chassis_move_status", "sub"),
    # --- Publishers ---
    TopicSpec("policy_action", JointState, "/policy/joint_states", "pub"),
    TopicSpec("policy_gripper_left", GripperStatus, "/policy/left_gripper_status", "pub"),
    TopicSpec("policy_gripper_right", GripperStatus, "/policy/right_gripper_status", "pub"),
    TopicSpec("chassis_move_command", ChassisMoving, "/agent/chassis_move_command", "pub"),
]


def ensure_connected(func):
    """Decorator to verify that the robot is connected before executing a method.

    Raises:
        RobotDeviceNotConnectedError: If the robot is not connected.
    """

    def wrapper(self, *args, **kwargs):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"{self.__class__.__name__} is not connected.")
        return func(self, *args, **kwargs)

    return wrapper


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[str, float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    return safe_goal_positions


class RosNode(Node):
    """
    RosNode encapsulates basic ROS2 communication:
    - topic subscription
    - topic publishing
    - safe startup/shutdown in multi-threaded context
    """

    def __init__(self, node_name: str = "ros_robot_node"):
        # Initialize the rclpy context only once globally
        if not rclpy.ok():
            rclpy.init()

        super().__init__(node_name)
        self._lock = threading.Lock()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._running = True
        self._spin_thread.start()
        self.recording = False
        self.recording_last = False
        self.rerecord = False
        self.quit_recording = False
        self.policy_control = False
        self.last_chassis_status = None
        self.topics = {}  # store topic by key, both sub and pub
        # --- Create Subscriptions & Publishers ---
        num_sub, num_pub = 0, 0
        for spec in TOPICS:
            self.topics[spec.key] = None
            if spec.topic_type == "sub":
                callback = self._make_callback(spec.key)
                # Use rclpy.qos.qos_profile_sensor_data for camera topics for better compatibility
                qos = rclpy.qos.qos_profile_sensor_data if "camera" in spec.key else COMMON_QOS
                self.create_subscription(
                    spec.msg_type,
                    spec.topic_name,
                    callback,
                    qos_profile=qos,
                )
                num_sub += 1
                print(f"[bright_yellow][Sub] {spec.key}: {spec.topic_name}")
            if spec.topic_type == "pub":
                pub_qos = CHASSIS_COMMAND_QOS if spec.key == "chassis_move_command" else COMMON_QOS
                self.topics[spec.key] = self.create_publisher(
                    spec.msg_type,
                    spec.topic_name,
                    qos_profile=pub_qos,
                )  # store publisher object
                num_pub += 1
                print(f"[bright_blue][Pub] {spec.key}: {spec.topic_name}")

        print(f"[bright_green]Initialized {num_sub} subscribers and {num_pub} publishers, node is running.")

    # --- ROS Spin Thread ---
    def _spin(self):
        """Keep ROS2 spinning in background thread"""
        while rclpy.ok() and self._running:
            rclpy.spin_once(self, timeout_sec=0.05)

    # --- Callbacks ---
    def _make_callback(self, key: str):
        def callback(msg):
            with self._lock:
                self.topics[key] = msg  # update latest message for the sub topic
                if key == "switch_status":
                    self.recording_last = self.recording
                    self.recording = msg.recording
                    self.rerecord = msg.rerecord
                    self.quit_recording = msg.quit
                    self.policy_control = msg.model_control

        return callback

    # --- Data Access ---
    def read(self, key: str, poll_interval: float = 0.01):
        """Get latest data for a subscribed topic, waiting until the first message arrives."""
        # Check if key exists and the topic is a subscription
        if key not in self.topics:
            raise KeyError(f"Topic '{key}' not found in ROS node.")
        if key not in [spec.key for spec in TOPICS if spec.topic_type == "sub"]:
            raise KeyError(f"Topic '{key}' is not a subscriber topic.")
        while True:
            with self._lock:
                data = self.topics.get(key, None)
            if data is not None:
                return data
            time.sleep(poll_interval)

    # --- Send message by topic ---
    def send(self, key: str, msg_dict: dict):
        """Publish a message to a topic."""
        # Check if key exists and the topic is a publisher
        if key not in self.topics:
            raise KeyError(f"Topic '{key}' not found in ROS node.")
        if not isinstance(self.topics[key], rclpy.publisher.Publisher):
            raise KeyError(f"Topic '{key}' is not a publisher topic.")
        # Construct and publish the message
        msg = [spec.msg_type for spec in TOPICS if spec.key == key][0]()
        msg.header.stamp = self.get_clock().now().to_msg()
        if isinstance(msg, JointState):
            msg.name = list(msg_dict.keys())
            msg.position = [float(val) for val in msg_dict.values()]
        if isinstance(msg, GripperStatus):
            msg.current_status = list(msg_dict.values())[0]
        if isinstance(msg, ChassisMoving):
            msg.pub_source = msg_dict.get("pub_source", 1)
            msg.chassis_move_direction = msg_dict.get("chassis_move_direction", 0)
            msg.chassis_status = msg_dict.get("chassis_status", 0)
        with self._lock:
            self.topics[key].publish(msg)  # publish the message

    def start(self):
        """Start spinning in a background thread if not already running.

        This method is idempotent and safe to call multiple times. If the
        spin thread is already running, it does nothing. If rclpy is not
        initialized (e.g., after a previous shutdown), it re-initializes it.
        """
        # If already running and thread alive, no-op
        if (
            getattr(self, "_running", False)
            and getattr(self, "_spin_thread", None) is not None
            and self._spin_thread.is_alive()
        ):
            return

        # Ensure rclpy is initialized
        if not rclpy.ok():
            rclpy.init()

        # Start background spin thread
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    # --- Shutdown ---
    def stop(self):
        """Stop spinning and shut down node"""
        # If already stopped, no-op
        if not getattr(self, "_running", False):
            return

        self._running = False
        # Join the spin thread if it exists
        t = getattr(self, "_spin_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=1.0)  # wait for spin thread to finish

        # Only shutdown rclpy if it's currently running
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"Exception during rclpy shutdown: {e}")
        print("ROS node stopped.")

    def is_running(self) -> bool:
        return (
            getattr(self, "_running", False)
            and getattr(self, "_spin_thread", None) is not None
            and self._spin_thread.is_alive()
        )

    def is_ready(self) -> bool:
        """Check if the node has received initial messages for all subscribed topics."""
        for spec in TOPICS:
            if spec.topic_type == "sub" and spec.required:
                with self._lock:
                    if self.topics.get(spec.key, None) is None:
                        return False
        return True

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Wait until the node has received initial messages for all subscribed topics."""
        sub_keys = [spec.key for spec in TOPICS if spec.topic_type == "sub" and spec.required]
        start_time = time.monotonic()

        def render_status_table(elapsed: float) -> Table:
            with self._lock:
                statuses = {key: (self.topics.get(key, None) is not None) for key in sub_keys}

            table = Table(
                title=f"Waiting for ROS subscribers to be ready... {elapsed:.1f}s / {timeout:.1f}s",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Topic", style="white", no_wrap=True)
            table.add_column("Status", no_wrap=True)

            for key in sub_keys:
                is_ok = statuses[key]
                table.add_row(key, "[green]OK[/green]" if is_ok else "[red]--[/red]")

            return table

        with Live(render_status_table(0.0), refresh_per_second=10, transient=False) as live:
            while time.monotonic() - start_time < timeout:
                elapsed = time.monotonic() - start_time
                live.update(render_status_table(elapsed), refresh=True)

                if self.is_ready():
                    print("ROS node is ready with initial data on all subscribed topics.")
                    return True

                time.sleep(0.1)

            elapsed = time.monotonic() - start_time
            live.update(render_status_table(elapsed), refresh=True)

        print(f"Timeout waiting for ROS node to be ready after {elapsed:.1f}s.")
        return False


def show_numpy_img_realtime(img_array: np.ndarray, window_name="Realtime Image"):
    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, img_array)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        return False

    return True


def display_numpy_images(images: dict[str, np.ndarray], window_name="Realtime Images", show_depth=False):
    def normalize_to_u8(gray_like: np.ndarray) -> np.ndarray:
        arr = gray_like.astype(np.float32)
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max <= arr_min:
            return np.zeros_like(arr, dtype=np.uint8)
        arr = (arr - arr_min) * (255.0 / (arr_max - arr_min))
        return arr.astype(np.uint8)

    def to_rgb_for_display(img: np.ndarray, is_depth: bool) -> np.ndarray:
        # Depth images are visualized as grayscale repeated to 3 channels.
        if is_depth:
            if img.ndim == 2:
                gray = img
            elif img.shape[-1] >= 1:
                gray = img[..., 0]
            else:
                gray = img.squeeze()
            gray_u8 = normalize_to_u8(gray)
            return np.repeat(gray_u8[..., None], 3, axis=2)

        # RGB images: keep first 3 channels and convert RGB->BGR for cv2 display.
        if img.ndim == 2:
            gray_u8 = normalize_to_u8(img)
            rgb = np.repeat(gray_u8[..., None], 3, axis=2)
        elif img.shape[-1] >= 3:
            rgb = img[..., :3].astype(np.uint8, copy=False)
        elif img.shape[-1] == 2:
            ch0 = normalize_to_u8(img[..., 0])
            ch1 = normalize_to_u8(img[..., 1])
            rgb = np.stack([ch0, ch1, ch0], axis=2)
        else:
            gray_u8 = normalize_to_u8(img[..., 0])
            rgb = np.repeat(gray_u8[..., None], 3, axis=2)

        return rgb[..., ::-1]

    rgb_list = []
    depth_list = []

    for key, img in images.items():
        if img is None:
            continue
        is_depth = ("depth" in key) or (img.ndim == 3 and img.shape[-1] == 2)
        if is_depth and not show_depth:
            continue
        disp = to_rgb_for_display(img, is_depth=is_depth)
        if is_depth:
            depth_list.append(disp)
        else:
            rgb_list.append(disp)

    rows = []
    if rgb_list:
        rows.append(np.hstack(rgb_list))
    if depth_list:
        rows.append(np.hstack(depth_list))

    if not rows:
        return True

    if len(rows) == 2 and rows[0].shape[1] != rows[1].shape[1]:
        max_w = max(rows[0].shape[1], rows[1].shape[1])
        padded_rows = []
        for row in rows:
            pad_w = max_w - row.shape[1]
            if pad_w > 0:
                row = np.pad(row, ((0, 0), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            padded_rows.append(row)
        rows = padded_rows

    combined_img = rows[0] if len(rows) == 1 else np.vstack(rows)

    cv2.imshow(window_name, combined_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        return False

    return True


@dataclass
class RosRobotConfig:
    # Robot type (e.g. "ros_robot")
    type: str = "ros_robot"
    # Enable the robot to control the left arm, right arm, or both arms.
    enable: str = "bimanual"  # "left", "right", or "bimanual"
    # Default reset pose. Order must match `self.motors`.
    reset_pose: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0, 0.0]
    )
    # Robot cameras configuration overriding the default robot cameras.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": OpenCVCameraConfig(camera_index="/dev/video2", fps=30, width=640, height=480, fourcc="MJPG"),
            "left": OpenCVCameraConfig(camera_index="/dev/video4", fps=30, width=640, height=480, fourcc="MJPG"),
            "right": OpenCVCameraConfig(camera_index="/dev/video0", fps=30, width=640, height=480, fourcc="MJPG"),
        }
    )

    def __post_init__(self):
        if self.type != "ros_robot":
            raise ValueError(f"Invalid robot type '{self.type}' for RosRobotConfig. Expected 'ros_robot'.")
        if not isinstance(self.reset_pose, list) or len(self.reset_pose) != 16:
            raise ValueError("reset_pose must be a list of 16 float values corresponding to motor positions.")
        if not isinstance(self.cameras, dict):
            raise ValueError("cameras must be a dictionary mapping camera names to CameraConfig objects.")
        for cam_name, cam_cfg in self.cameras.items():
            if not isinstance(cam_cfg, CameraConfig):
                # Construct specific CameraConfig objects based on dict parameters if needed
                if cam_cfg.get("type") == "opencv":
                    kargs = cam_cfg.copy()
                    kargs.pop("type")
                    self.cameras[cam_name] = OpenCVCameraConfig(**kargs)
                elif cam_cfg.get("type") == "realsense":
                    kargs = cam_cfg.copy()
                    kargs.pop("type")
                    self.cameras[cam_name] = IntelRealSenseCameraConfig(**kargs)
                else:
                    raise ValueError(f"Unsupported camera config type for camera '{cam_name}'.")
        assert len(self.cameras) > 0, "At least one camera must be configured for the robot."


class RosRobot(Robot):
    def __init__(self, config: RosRobotConfig):
        self.name = self.__class__.__name__
        self.config = config
        self.enable = config.enable  # "left", "right", or "bimanual"
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.motors = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "LG", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "RG"]
        self.n_motors = len(self.motors)
        assert self.n_motors % 2 == 0, "Expected even number of motors for left/right arm split."
        self.task = None  # Task string sent to the policy server
        self.node = RosNode()
        self.node.wait_until_ready(timeout=np.inf)
        self._last_joint_state = None
        # ROS 2 action client for multi-waypoint P2P arm motion
        self.action_client = ActionClient(self.node, MoveToJoints, "follower/move_to_joints")
        self.action_progress = 0.0  # Progress of the current action (0.0 to 1.0)
        # Auto connect on initialization
        self.connect()

    def set_task(self, task: str):
        self.task = task
        print(f"[bright_yellow]Set {self.name} task to: {task}[/bright_yellow]")

    @property
    def motor_features(self) -> dict[str, type]:
        motors = (
            self.motors
            if self.enable == "bimanual"
            else (self.motors[: self.n_motors // 2] if self.enable == "left" else self.motors[self.n_motors // 2 :])
        )
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": [f"states/{motor}.pos" for motor in motors],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": [f"teleop/{motor}.pos" for motor in motors],
            },
            "policy_action": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": [f"policy/{motor}.pos" for motor in motors],
            },
            "complementary_info.action_vel": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": [f"teleop/{motor}.vel" for motor in motors],
            },
            "complementary_info.action_fix": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": [f"teleop/{motor}.fix" for motor in motors],
            },
        }

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "dtype": "video",
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def lift_height(self) -> dict[str, type]:
        return {
            "complementary_info.lift_height": {
                "dtype": "float32",
                "shape": (1,),
            }
        }

    @property
    def control_mode(self) -> dict[str, type]:
        return {
            "complementary_info.control_mode": {
                "dtype": "float32",
                "shape": (1,),
            }
        }

    @cached_property
    def features(self) -> dict[str, type | tuple]:
        return {**self.motor_features, **self.camera_features, **self.control_mode}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self.motor_features

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def is_connected(self) -> bool:
        return self.node.is_running() and all(camera.is_connected for camera in self.cameras.values())

    def connect(self) -> None:
        print(f"Connecting to <{self.__class__.__name__}>...")
        if self.is_connected:
            print(f"[green]<{self.__class__.__name__}> is already connected.[/green]")
            return
        # Start ROS2 background spinning
        for _, camera in self.cameras.items():
            camera.connect()
        self.node.start()
        print(f"[green]<{self.__class__.__name__}> connected.[/green]")
        self.warmup()

    def warmup(self, duration: float = 3.0) -> None:
        """Observe camera images and subscribed topics until they are all ready to provide data.
        This method makes sure the camera images and joint states are available and avoids getting stuck
        when the first call of `capture_observation` or `read_motor_state` is made.
        """
        print(f"Warming up <{self.__class__.__name__}>...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            if self.is_connected and self.node.is_ready():
                # Try reading data to ensure topics are active
                obs = self.capture_observation()
                if all(value is not None for value in obs.values()):
                    print(f"[green]<{self.__class__.__name__}> is warmed up and ready.[/green]")
                    break
            time.sleep(0.1)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    @property
    def pressed_start(self):
        return not self.node.recording_last and self.node.recording

    @property
    def pressed_stop(self):
        return self.node.recording_last and not self.node.recording

    def refresh_last(self):
        self.node.recording_last = self.node.recording

    @property
    def pressed_rerecord(self):
        return self.node.rerecord

    @property
    def pressed_quit(self):
        return self.node.quit_recording

    @property
    def policy_control(self):
        return self.node.policy_control

    @ensure_connected
    def get_control_mode(self):
        return {"complementary_info.control_mode": np.array([self.policy_control], dtype=np.float32)}

    @ensure_connected
    def get_lift_height(self):
        # Read lift height
        #! This method is STUCK!
        #! DO NOT call it in the main thread to avoid blocking.
        data = self.node.read("lift_height")
        return {
            "complementary_info.lift_height": np.array(
                [
                    data.lift_height + LIFT_HEIGHT_OFFSET
                    if data is not None
                    else LIFT_DEFAULT_HEIGHT + LIFT_HEIGHT_OFFSET
                ],
                dtype=np.float32,
            )
        }

    @ensure_connected
    def arm_is_moving(self, motion_threshold: float = 0.01) -> bool:
        """Determine if the arm is currently moving based on the change of the joints position."""
        current_state = self.read_motor_state()
        if self._last_joint_state is None:
            self._last_joint_state = current_state.copy()
            return False

        diff = np.abs(current_state - self._last_joint_state)
        self._last_joint_state = current_state.copy()
        return bool(np.max(diff) > motion_threshold)

    @ensure_connected
    def reset_motion_detection(self):
        self._last_joint_state = None

    @ensure_connected
    def read_motor_state(self, teleop: bool = False) -> np.ndarray:
        # Read motor position
        joints = self.node.read("teleop_action" if teleop else "joint_states").position  # (14,)
        LJ = joints[:7]  # 7 joints for the left arm
        RJ = joints[7:]  # 7 joints for the right arm
        assert len(LJ) == 7 and len(RJ) == 7
        # Read gripper status
        LG = self.node.read("left_gripper_status").current_status
        RG = self.node.read("right_gripper_status").current_status
        # Assemble motor states into a single array in the order of self.motors
        states = list(LJ) + [LG] + list(RJ) + [RG]
        assert len(states) == self.n_motors, f"Expected {self.n_motors} motor states, got {len(states)}"

        return np.array(states, dtype=np.float32)

    @ensure_connected
    def capture_observation(self) -> dict[str, torch.Tensor]:
        observation = {"task": self.task}

        # Read motor position and gripper status as part of the observation
        states = self.read_motor_state(teleop=False)
        # Slice the states according to the enabled arm(s)
        states = (
            states[: self.n_motors // 2]
            if self.enable == "left"
            else states[self.n_motors // 2 :]
            if self.enable == "right"
            else states
        )
        observation["observation.state"] = torch.from_numpy(states)

        # Read camera images concurrently
        def read_camera(item):
            cam_name, camera = item
            image = camera.async_read()
            return cam_name, torch.from_numpy(image)

        # Concurrently read camera images and store in observation dict
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            results = executor.map(read_camera, self.cameras.items())

        for cam_name, image in results:
            observation[f"observation.images.{cam_name}"] = image

        return observation

    @ensure_connected
    def get_teleop_action(self):
        """
        Get the action from the teleop controller.
        """
        action = self.read_motor_state(teleop=True)
        assert len(action) == len(self.motors), f"Expected {len(self.motors)} motor states, got {len(action)}"
        return {
            "action": torch.from_numpy(action.astype(np.float32)),
            "complementary_info.action_vel": torch.zeros(len(self.motors), dtype=torch.float32),
            "complementary_info.action_fix": torch.zeros(len(self.motors), dtype=torch.float32),
        }

    @staticmethod
    def _to_action_array(action: np.ndarray | torch.Tensor | list[float]) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32).reshape(-1)

    @ensure_connected
    def send_action(self, action: np.ndarray | torch.Tensor | list[float]) -> dict[str, torch.Tensor]:
        """Send the action command to the robot via ROS topics.
        Args:
            action: Array-like of motor commands.
        """
        assert self.is_connected, f"{self.__class__.__name__} is not connected."
        action_dim = self.n_motors if self.enable == "bimanual" else self.n_motors // 2
        action = action[:action_dim]  # Ensure action length does not exceed number of motors
        action = self._to_action_array(action)  # Convert to numpy array safely
        actual_action = torch.from_numpy(action)  # For returning in the output dict

        # Complete action with static states if only one arm is enabled.
        if self.enable == "left" or self.enable == "right":
            assert len(action) == action_dim, (
                f"Expected {action_dim} actions for the {self.enable} arm, got {len(action)}"
            )
            states = self.read_motor_state(teleop=False)
            # Complete action with current states of the other arm
            states = states[action_dim:] if self.enable == "left" else states[:action_dim]
            action = np.concatenate([action, states]) if self.enable == "left" else np.concatenate([states, action])

        # Finally, action should have the same length as self.motors
        self.send_motor_state(action)

        assert len(actual_action) == action_dim
        return {
            "action": actual_action,
            "complementary_info.action_vel": torch.zeros(len(self.motors), dtype=torch.float32),
            "complementary_info.action_fix": torch.zeros(len(self.motors), dtype=torch.float32),
        }

    @ensure_connected
    def send_motor_state(self, states: np.ndarray) -> bool:
        """Directly send motor states as action command to the robot."""
        joint_names = [
            "joint_handle_L1",
            "joint_handle_L2",
            "joint_handle_L3",
            "joint_handle_L4",
            "joint_handle_L5",
            "joint_handle_L6",
            "joint_handle_L7",
            "joint_handle_R1",
            "joint_handle_R2",
            "joint_handle_R3",
            "joint_handle_R4",
            "joint_handle_R5",
            "joint_handle_R6",
            "joint_handle_R7",
        ]
        try:
            assert len(states) == len(self.motors), f"Expected {len(self.motors)} motor states, got {len(states)}"
            joint_vals = list(states[:7]) + list(states[8:15])  #! Careful: exclude the gripper
            joint_dict = dict(zip(joint_names, joint_vals, strict=True))
            self.node.send("policy_action", joint_dict)
            LG = int(states[7] > 0.5)
            RG = int(states[15] > 0.5)
            self.node.send("policy_gripper_left", {"value": LG})
            self.node.send("policy_gripper_right", {"value": RG})
            return True
        except AssertionError as e:
            print(f"[red]{e}[/red]")
            return False

    @ensure_connected
    def switch_gripper(self, side: str):
        assert side in ["left", "right"], f"Invalid gripper side: {side}"
        message = self.node.read(f"{side}_gripper_status")
        assert message is not None, f"Gripper status for {side} side is not available."
        current_status = message.current_status
        assert current_status in [0, 1], f"Unexpected {side} gripper current status value: {current_status}"
        self.node.send(f"policy_gripper_{side}", {"value": 1 - current_status})

    @ensure_connected
    def reset(self, target_position: list[float] | None = None, timeout: float = 5.0) -> None:
        """Move the robot to a user-defined reset pose.

        Args:
            target_position: Optional 16-dim joint target in the order of `self.motors`.
                If omitted, `self.config.reset_pose` is used.
            timeout: Maximum reset duration in seconds.
        """
        # Wait until getting current joint states
        while self.node.read("joint_states") is None:
            time.sleep(0.1)

        target_position = list(target_position or self.config.reset_pose)
        if len(target_position) != len(self.motors):
            raise ValueError(
                f"Expected reset target of length {len(self.motors)}, got {len(target_position)}: {target_position}"
            )

        print(f"[yellow]Reset to target position: {target_position}")
        is_success = self.set_arms(target_position, max_duration_s=timeout)
        if not is_success:
            print("[bright_red]Reset failed.[/bright_red]")
        else:
            print("[bright_green]Reset successful.[/bright_green]")

    @ensure_connected
    def set_arms(
        self,
        target_pos: list[float],
        kp: float = 1.0,
        max_step: float = 0.12,
        tol: float = 0.06,
        dt: float = 0.05,
        max_duration_s: float = 5.0,
    ) -> bool:
        """Closed-loop move to target joint positions.

        Args:
            target_positions: List of target joint positions (radians) for all 16 joints.
            kp: Proportional gain.
            max_step: Max radians per joint per iteration.
            tol: Convergence tolerance (rad) applied to targeted joints.
            dt: Control loop period (s).
            max_duration_s: Safety timeout for the motion (s).

        Returns:
            True if the motion is successful, False otherwise.
        """

        start_t = time.monotonic()
        while True:
            current_pos = self.read_motor_state(teleop=False).tolist()
            # Check convergence on targeted joints
            if all(abs(target_pos[i] - current_pos[i]) <= tol for i in range(len(target_pos))):
                print(f"[bright_green]Target position reached in {time.monotonic() - start_t:.2f} s.[/bright_green]")
                return True
            # Check if timeout
            if (time.monotonic() - start_t) > max_duration_s:
                print(f"[bright_red]Target position not reached within {max_duration_s} s timeout![/bright_red]")
                return False
            # Compute bounded incremental step
            cmd_pos = []
            pos_err = kp * (np.asarray(target_pos) - np.asarray(current_pos))
            err_max_abs = np.abs(pos_err).max()
            scale = min(1.0, max_step / err_max_abs) if err_max_abs > 0 else 1.0
            cmd_pos = (np.asarray(current_pos) + pos_err * scale).tolist()
            cmd_pos[7] = target_pos[7]  # gripper left
            cmd_pos[15] = target_pos[15]  # gripper right
            self.send_motor_state(np.array(cmd_pos))
            time.sleep(dt)

    @ensure_connected
    def move_arms_by_action(self, target_positions: list[list[float]], timeout_sec: float = 60.0) -> bool:
        """Drive both arms through a sequence of joint waypoints via the follower/move_to_joints action.

        Args:
            target_positions: list of target joint positions, each 16-dim in the order of `self.motors`
                (7 left joints + 1 left gripper + 7 right joints + 1 right gripper).
            timeout_sec: max wait time for the action server and the final result.

        Returns:
            True if the action server reported success, False otherwise (server missing, goal rejected,
            no result, or the action itself reported failure).

        Side effect:
            `self.action_progress` is updated in real time (0.0 to 1.0) from the action's
            `feedback.progress_percentage` (0-100). It resets to 0.0 on entry and is set
            to 1.0 on successful completion.
        """
        self.action_progress = 0.0

        def _on_feedback(feedback_msg):
            self.action_progress = float(feedback_msg.feedback.progress_percentage) / 100.0

        if not self.action_client.wait_for_server(timeout_sec=timeout_sec):
            print(f"[bright_red]follower/move_to_joints action server not available after {timeout_sec}s[/bright_red]")
            return False

        goal_msg = MoveToJoints.Goal()
        for pos in target_positions:
            wp = JointsWaypoint()
            wp.target_positions = [float(v) for v in pos]
            goal_msg.waypoints.append(wp)

        send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=_on_feedback)
        rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=timeout_sec)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            print("[bright_red]follower/move_to_joints goal was REJECTED[/bright_red]")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout_sec)
        result_response = result_future.result()
        if result_response is None:
            return False
        success = bool(result_response.result.success)
        if success:
            self.action_progress = 1.0
        return success

    @ensure_connected
    def move_chassis(self, direction: str, timeout: float = 60.0):
        """Move the chassis of the robot in the specified direction"""

        assert direction in ["forward", "backward", "left", "right"], f"Invalid direction: {direction}"

        # chassis move command publish
        command = {
            "pub_source": 1,
            "chassis_move_direction": CHASSIS_DIRECTION_MAP[direction],
            "chassis_status": 0,
        }
        print(f"Sending chassis move command: {direction}")
        self.node.send("chassis_move_command", command)

        # waiting for chassis move feedback: MOVING → SUCCESS
        start_time = time.monotonic()
        check_interval = 0.05

        while True:
            if time.monotonic() - start_time > timeout:
                raise RobotChassisMoveError(f"Chassis move {direction} timeout.")

            time.sleep(check_interval)

            msg = self.node.read("chassis_move_feedback")
            if msg is None:
                print("[yellow]Chassis moving status feedback not detected...")
                continue

            current_status = msg.chassis_status

            if (
                self.node.last_chassis_status == CHASSIS_STATUS_MAP["moving"]
                and current_status == CHASSIS_STATUS_MAP["success"]
            ):
                print(f"[green]Chassis move {direction} completed successfully.")
                break

            # move fail
            if current_status == CHASSIS_STATUS_MAP["failure"]:
                raise RobotChassisMoveError(f"Chassis move {direction} failed!")

            self.node.last_chassis_status = current_status

    @ensure_connected
    def disconnect(self):
        for camera in self.cameras.values():
            camera.disconnect()
        self.node.stop()
        print(f"{self} disconnected.")

    @ensure_connected
    def stop(self):
        """alias for disconnect() to match the Robot interface"""
        self.disconnect()

    @ensure_connected
    def move_to_location(self, value, **kwargs):
        """Move to a specified location."""
        print(f"[RosRobot][NotImplemented] move_to_location: {value}")

    @ensure_connected
    def set_torso_height(self, value, **kwargs):
        print(f"[RosRobot][NotImplemented] set_torso_height: {value}m")

    @ensure_connected
    def set_joint_angles(self, value, **kwargs) -> bool:
        t0 = time.monotonic()
        timeout = kwargs.get("timeout", 5.0)
        epsilon = kwargs.get("epsilon", 0.06)
        while time.monotonic() - t0 < timeout:
            state = self.read_motor_state(teleop=False).tolist()
            # Check convergence on targeted joints
            if all(abs(value[i] - state[i]) <= epsilon for i in range(len(value))):
                return True
            # Check if timeout
            if (time.monotonic() - t0) > timeout:
                print(f"[bright_red]Target position not reached within {timeout} s timeout![/bright_red]")
                return False
            # Compute bounded incremental step
            cmd_pos = []
            pos_err = np.asarray(value) - np.asarray(state)
            err_max_abs = np.abs(pos_err).max()
            scale = min(1.0, epsilon / err_max_abs) if err_max_abs > 0 else 1.0
            cmd_pos = (np.asarray(state) + pos_err * scale).tolist()
            cmd_pos[7] = value[7]  # gripper left
            cmd_pos[15] = value[15]  # gripper right
            self.send_motor_state(np.array(cmd_pos))
            time.sleep(0.033)  # ~30Hz control loop
        return True

    @ensure_connected
    def current_location(self):
        """Get the current location of the robot."""
        print(f"[RosRobot][NotImplemented] current_location")
        return "A"

    @ensure_connected
    def get_torso_height(self):
        """Get the current torso height of the robot."""
        print(f"[RosRobot][NotImplemented] get_torso_height")
        return 0.75

    @ensure_connected
    def get_joint_angles(self):
        """Get the current joint angles of the robot."""
        print(f"[RosRobot][NotImplemented] get_joint_angles")
        return [0.0] * 16


def test_robot(robot: RosRobot, num_iterations: int = 100, rerun_url: str = ""):
    def _tensor_meta(v):
        if isinstance(v, torch.Tensor):
            return str(tuple(v.shape)), str(v.dtype)
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            return str(tuple(v.shape)), str(v.dtype)
        return "-", str(type(v).__name__)

    def _value_preview(v, max_items: int = 8):
        if isinstance(v, torch.Tensor):
            tensor = v.detach().cpu()
            if tensor.ndim >= 2:
                tensor_float = tensor.float()
                return (
                    f"tensor(min={tensor_float.min().item():.4f}, "
                    f"max={tensor_float.max().item():.4f}, "
                    f"mean={tensor_float.mean().item():.4f})"
                )

            flat = tensor.flatten()
            n = min(max_items, flat.numel())
            values = ", ".join(f"{flat[i].item():.4f}" for i in range(n))
            suffix = " ..." if flat.numel() > n else ""
            return f"[{values}{suffix}]"

        if isinstance(v, np.ndarray):
            if v.ndim >= 2:
                arr = v.astype(np.float32, copy=False)
                return f"array(min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f})"

            flat = v.reshape(-1)
            n = min(max_items, flat.size)
            values = ", ".join(f"{float(flat[i]):.4f}" for i in range(n))
            suffix = " ..." if flat.size > n else ""
            return f"[{values}{suffix}]"

        return str(v)

    def _render_live_view(obs_ms: float, act_ms: float, obs: dict, act: dict):
        perf_table = Table(title="RosRobot Test Metrics", header_style="bold cyan")
        perf_table.add_column("Method", style="white", no_wrap=True, justify="right")
        perf_table.add_column("Delay", style="green", no_wrap=True, justify="right")
        perf_table.add_row("Capture Observation", f"{obs_ms:.2f} ms")
        perf_table.add_row("Get Teleop Action", f"{act_ms:.2f} ms")

        data_table = Table(title="Tensor Overview", header_style="bold magenta")
        data_table.add_column("Key", overflow="fold")
        data_table.add_column("Shape", no_wrap=True)
        data_table.add_column("DType", no_wrap=True)
        data_table.add_column("Value", overflow="fold")

        for key, value in obs.items():
            shape, dtype = _tensor_meta(value)
            data_table.add_row(key, shape, dtype, _value_preview(value))

        for key, value in act.items():
            shape, dtype = _tensor_meta(value)
            data_table.add_row(key, shape, dtype, _value_preview(value))

        return Panel(Group(progress, perf_table, data_table), title="ROS Robot Runtime", border_style="bright_blue")

    def _delay_stats(delays: list[float]) -> tuple[float, float, float, float]:
        arr = np.asarray(delays, dtype=np.float64)
        return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())

    progress = Progress(
        TextColumn("[bold blue]Test Progress"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task("ros_robot_test", total=num_iterations)
    obs_delays_ms: list[float] = []
    act_delays_ms: list[float] = []

    with Live(refresh_per_second=4, transient=False) as live:
        for _ in range(num_iterations):
            start = time.perf_counter()
            obs = robot.capture_observation()
            obs_ms = (time.perf_counter() - start) * 1e3
            start = time.perf_counter()
            act = robot.get_teleop_action()
            act_ms = (time.perf_counter() - start) * 1e3

            obs_delays_ms.append(obs_ms)
            act_delays_ms.append(act_ms)
            progress.update(task_id, advance=1)

            live.update(_render_live_view(obs_ms, act_ms, obs, act), refresh=True)

    obs_min, obs_max, obs_mean, obs_std = _delay_stats(obs_delays_ms)
    act_min, act_max, act_mean, act_std = _delay_stats(act_delays_ms)

    stats_table = Table(title="Delay Statistics (ms)", header_style="bold green")
    stats_table.add_column("Method", style="white")
    stats_table.add_column("Min", justify="right")
    stats_table.add_column("Max", justify="right")
    stats_table.add_column("Mean ± Std", justify="right")
    stats_table.add_row("Capture Observation", f"{obs_min:.2f}", f"{obs_max:.2f}", f"{obs_mean:.2f} ± {obs_std:.2f}")
    stats_table.add_row("Get Teleop Action", f"{act_min:.2f}", f"{act_max:.2f}", f"{act_mean:.2f} ± {act_std:.2f}")
    print(stats_table)

    print("Test completed. Disconnecting robot...")
    robot.disconnect()
    print("Robot disconnected. Exiting.")


@dataclass
class Config:
    robot: RosRobotConfig
    rerun_url: str = ""


@draccus.wrap()
def main(config: Config):
    print(config)
    robot = RosRobot(config.robot)
    robot.set_task("do nothing")
    robot.connect()
    test_robot(robot, num_iterations=100, rerun_url=config.rerun_url)


if __name__ == "__main__":
    main()
