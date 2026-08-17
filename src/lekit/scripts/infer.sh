#!/bin/bash


task="${1:-"pick the tape and place it on the pad."}"

cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
           wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" 

# rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
rename_map='{}'

rm -rf "/home/sorel/.cache/huggingface/lerobot/sorel/eval_so101"


python scripts/tools/infer.py  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="$cameras" \
  --robot.id=my_awesome_follower_arm \
  --display_data=true \
  --dataset.repo_id="sorel/eval_so101" \
  --dataset.single_task="$task" \
  --dataset.rename_map="$rename_map" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=3600 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --policy.host="127.0.0.1" \
  --policy.port=8001 \
  --policy.chunk_size_threshold=0.5
