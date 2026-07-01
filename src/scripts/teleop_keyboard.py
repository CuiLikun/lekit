#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keyboard teleoperation for so101_follower using IK pipeline."""

import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM1",
        id="my_awesome_follower_arm",
        cameras={
            "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            "wrist": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
        },
    )
    teleop_config = KeyboardEndEffectorTeleopConfig(use_gripper=True)

    robot = SO101Follower(robot_config)
    teleop_device = KeyboardEndEffectorTeleop(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo
    kinematics_solver = RobotKinematics(
        urdf_path="./examples/phone_to_so100/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline: keyboard delta action -> ee pose -> joints
    keyboard_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapDeltaActionToRobotActionStep(position_scale=0.01),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()
    teleop_device.connect()

    init_rerun(session_name="keyboard_so101_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting keyboard teleop. Use arrow keys + Shift for movement, Ctrl for gripper...")
    print("Controls:")
    print("  Arrow keys: X/Y movement")
    print("  Shift/Shift_R: Z movement")
    print("  Ctrl_R/Ctrl_L: gripper open/close")
    print("  S: success, R: rerecord, Q: quit")
    print("  ESC: disconnect")

    while True:
        t0 = time.perf_counter()

        robot_obs = robot.get_observation()
        keyboard_action = teleop_device.get_action()
        joint_action = keyboard_to_robot_joints_processor((keyboard_action, robot_obs))

        _ = robot.send_action(joint_action)

        log_rerun_data(observation=robot_obs, action=joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()