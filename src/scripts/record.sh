#!/bin/bash
# Usage: ./record.sh <dataset> [task]

dataset=$1

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset> [task]"
    exit 1
fi

task="${2:-"pick the tape and place it on the pad."}"


cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
    wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"

python -m src.scripts.lerobot_record \
    --robot.type="mock_robot" \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="$cameras" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id="sorel/${dataset}" \
    --dataset.num_episodes=50 \
    --dataset.single_task="$task" \
    --dataset.streaming_encoding=true \
    --dataset.push_to_hub=true