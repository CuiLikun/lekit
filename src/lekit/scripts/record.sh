#!/bin/bash
# Usage: ./record.sh <dataset> [task] [extra-args...]
#
# Record a LeRobot dataset driven by NVIDIA Isaac Teleop on a Meta Quest 3
# (or other CloudXR-capable VR headset) -- via the in-memory ``mock_robot``
# follower. The Quest 3 controller pose flows through the real ``XRController``
# in ``examples.isaac_teleop_to_so101``, which:
#
#   1. Auto-launches the CloudXR runtime and prints the workstation IP for
#      the headset to connect to.
#   2. Waits for the operator to don the headset and start streaming.
#   3. Per frame: reads the controller grip pose, runs the squeeze-to-engage
#      clutch + LeRobot's Cartesian IK pipeline, and commands the follower.

set -euo pipefail

dataset="${1:-}"
[[ -z "$dataset" ]] && {
    echo "Usage: $0 <dataset> [task] [extra-args...]" >&2
    exit 1
}
task="${2:-"pick up a tube from the pad"}"

repo_id="sorel/${dataset}"

exec python -m examples.isaac_teleop_to_so101.record \
    --robot.type="agx_arm" \
    --robot.id="piper_x" \
    --teleop.type="xr_controller" \
    --dataset.repo_id="$repo_id" \
    --dataset.num_episodes=50 \
    --dataset.single_task="$task" \
    --dataset.streaming_encoding=true \
    --dataset.push_to_hub=false
