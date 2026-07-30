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
#
# ``mock_robot`` absorbs the IK joint targets without moving anything physical
# (its motor dict is empty by default), so this is purely a smoke test of the
# CloudXR + Isaac Teleop + IK + recording stack -- useful for verifying the
# pipeline end-to-end on a headless workstation before going to real hardware.
# For real SO-101/SO-100 recording, see the ``lerobot-record`` /
# ``examples.isaac_teleop_to_so101.record`` entry-points directly.
#
# Env-var knobs (all optional):
#   ROBOT_ID=mock_arm                 mock follower id (passed to MockRobot)
#   ISAAC_TELEOP_TYPE=xr_controller   device on the Isaac path
#             | so101_leader          (xr_controller is the default -- Quest 3)
#   NUM_EPISODES=50                   episodes per run
#
# Positional ``[extra-args]`` are forwarded verbatim to the underlying Python
# entry-point, so callers can override individual flags (e.g.
# ``./record.sh foo bar --dataset.push_to_hub=true --display_data=true``).

set -euo pipefail

dataset="${1:-}"
[[ -z "$dataset" ]] && {
    echo "Usage: $0 <dataset> [task] [extra-args...]" >&2
    exit 1
}
task="${2:-"pick up a tube from the pad"}"
# Forward any positional >=3 to the underlying Python entry-point.
[[ $# -ge 1 ]] && shift  # drop <dataset>
[[ $# -ge 1 ]] && shift  # drop [task]
extra_args=("$@")

repo_id="sorel/${dataset}"
num_episodes="${NUM_EPISODES:-50}"
robot_id="${ROBOT_ID:-mock_arm}"
isaac_teleop_type="${ISAAC_TELEOP_TYPE:-xr_controller}"

exec python -m examples.isaac_teleop_to_so101.record \
    --robot.type=mock_robot \
    --robot.id="$robot_id" \
    --teleop.type="$isaac_teleop_type" \
    --dataset.repo_id="$repo_id" \
    --dataset.num_episodes="$num_episodes" \
    --dataset.single_task="$task" \
    --dataset.streaming_encoding=true \
    --dataset.push_to_hub=false \
    "${extra_args[@]}"