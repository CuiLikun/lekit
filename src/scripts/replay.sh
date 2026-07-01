#!/bin/bash

# source ~/orin_ws/ros2_ws/install/setup.bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# --episode can be a single number or a comma-separated list of numbers
# e.g., --episode="0,1,2" to replay episodes 0, 1, and 2 or --episode="0" to replay episode 0
# If not specified, all episodes will be replayed
# Example: Replay multiple episodes (comma-separated list)
python scripts/replay.py \
    --robot.type="replay_bot" \
    --robot.repo_id="sorel/record_1212" \
    --robot.episode="0,1,2" \
    --policy.host="127.0.0.1" \
    --policy.port=8000 \
    --policy.aggregate_fn_name="latest_only" \
    --fps=30 \
    --rerun_url="rerun+http://172.20.76.88:9876/proxy"
