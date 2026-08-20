#!/bin/bash
# record.sh
# --robot.cameras="{ base_camera: {type: stream, endpoint: 'tcp://0.0.0.0:5555', camera_name: base_camera, width: 640, height: 480, fps: 30}, hand_camera: {type: stream, endpoint: 'tcp://0.0.0.0:5555', camera_name: hand_camera, width: 640, height: 480, fps: 30}}" \

reset_joints="[-0.956, 1.903, 1.427, 1.368, -1.590, -0.290]"

endpoint="tcp://0.0.0.0:5555"
base_camera="base_camera: {type: stream, endpoint: '$endpoint', camera_name: base_camera, width: 640, height: 480, fps: 30}"
hand_camera="hand_camera: {type: stream, endpoint: '$endpoint', camera_name: hand_camera, width: 640, height: 480, fps: 30}"

uv run python -m examples.isaac_teleop_to_jaka.record \
    --robot.type=jaka_robot \
    --robot.id=jaka_arm \
    --robot.ip=192.168.1.31 \
    --robot.cameras="{$base_camera, $hand_camera}" \
    --robot.reset_joints="$reset_joints" \
    --teleop.type=xr_controller \
    --teleop.lock_pose=True \
    --teleop.thumbstick_deadband=0.15 \
    --teleop.thumbstick_angular_speed_rad_s=0.5 \
    --teleop.use_head_yaw=True \
    --teleop.operator_yaw_deg=0 \
    --reset_hold_s=1.0 \
    --dataset.repo_id="sorel/pick-cube" \
    --dataset.single_task="pick up the tube from the pad" \
    --dataset.fps=30 \
    --dataset.num_episodes=49 \
    --dataset.episode_time_s=9999 \
    --dataset.streaming_encoding=False \
    --dataset.push_to_hub=False \
    --rerun_url="rerun+http://127.0.0.1:9876/proxy" \
    "$@"
