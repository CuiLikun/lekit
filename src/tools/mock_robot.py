import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MockRobotConfig:
    # Robot type
    type: str = "mock_robot"
    # Select bimanual arms control
    enable: str = "bimanual"
    # Default reset pose for 16 joints
    reset_pose: list[float] = field(default_factory=lambda: [0.0] * 16)
    # Mock cameras
    cameras: dict = field(default_factory=dict)


class MockNode:
    def __init__(self):
        self.topics = {"switch_status": True}
        self.last_chassis_status = None

    def read(self, topic_name):
        return None

    def send(self, topic_name, value):
        pass


class MockCamera:
    def __init__(self, name):
        self.name = name

    def async_read(self):
        # Return a mock image tensor with 3 channels, 480 height, 640 width
        return torch.zeros((3, 480, 640), dtype=torch.uint8)

    def reset_async_read_stats(self):
        pass

    def get_async_read_stats(self):
        return {"async_read_stale_ratio": 0.0}


class MockRobot:
    """Mock robot class that mirrors RosRobot's API exactly for dry-run testing."""

    def __init__(self, config: MockRobotConfig):
        self.config = config
        self.is_connected = False
        self.task = "Mock robot chilling & loafing."
        self.node = MockNode()
        self.cameras = {name: MockCamera(name) for name in ["head", "left", "right"]}
        self.motors = list(range(16))
        self.policy_control = True
        self.connect()  # Auto-connect on initialization

    def connect(self):
        self.is_connected = True
        print("[MockRobot] Connected.")

    def disconnect(self):
        self.is_connected = False
        print("[MockRobot] Disconnected.")

    def stop(self):
        """alias for disconnect() to match the Robot interface"""
        self.disconnect()

    def move_to_location(self, value, **kwargs):
        """Move to a specified location."""
        print(f"[MockRobot] move_to_location: {value}")
        time.sleep(1.0)

    def set_torso_height(self, value, **kwargs):
        print(f"[MockRobot] set_torso_height: {value}m")
        time.sleep(1.0)

    def set_joint_angles(self, value, **kwargs) -> bool:
        print(f"[MockRobot] set_joint_angles: {value}")
        time.sleep(1.0)
        return True

    def current_location(self):
        """Get the current location of the robot."""
        return "A"

    def get_torso_height(self):
        """Get the current torso height of the robot."""
        return 0.75

    def get_joint_angles(self):
        """Get the current joint angles of the robot."""
        return [0.0] * 16

    def capture_observation(self) -> dict:
        obs = {"observation.state": torch.zeros((16,))}
        # Add mock camera image tensors
        for name in self.cameras:
            obs[f"observation.images.{name}"] = torch.zeros((3, 480, 640), dtype=torch.uint8)
        return obs

    def send_action(self, action: np.ndarray) -> dict:
        return {"action": action}

    def get_teleop_action(self) -> dict:
        return {"action": np.zeros((16,))}

    def get_control_mode(self) -> dict:
        return {"control_mode": "policy"}

    def move_arms_by_action(self, action: np.ndarray, **kwargs) -> dict:
        print(f"[MockRobot] move_arms_by_action: {action}")
        time.sleep(1.0)
        return {"status": "success"}
