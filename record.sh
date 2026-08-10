#!/bin/bash
    # --robot.cameras="{ hand: {type: intelrealsense, serial_number_or_name: '342522070741', width: 640, height: 480, fps: 30}}" \

uv run python -m examples.isaac_teleop_to_jaka.record \
    --robot.type=jaka_robot \
    --robot.ip=192.168.1.31 \
    --robot.id=jaka_arm \
    --teleop.type=xr_controller \
    --teleop.lock_pose=True \
    --teleop.use_head_yaw=True \
    --teleop.operator_yaw_deg=0 \
    --dataset.repo_id="sorel/pick-cube" \
    --dataset.single_task="Pick up the object" \
    --dataset.fps=30 \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=9999 \
    --dataset.reset_time_s=5 \
    --dataset.streaming_encoding=True \
    --dataset.push_to_hub=False \
    "$@"
