#!/bin/bash


task="${1:-"pick the tape and place it on the pad."}"

cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
           wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" 


python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="$cameras" \
    --robot.id=my_awesome_follower_arm \
    --task="$task" \
    --server_address=127.0.0.1:8080 \
    --policy_type="act" \
    --pretrained_name_or_path="sorel/act-so101-0330" \
    --policy_device=cuda \
    --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True