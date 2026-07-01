#!/bin/bash

usage="$0 <policy_type> <pretrained_name_or_path> [port]"

policy_type="$1"
ckpt="$2"
port="$3"

if [ -z "$ckpt" ] || [ -z "$policy_type" ]; then
    echo "Usage: $usage"
    exit 1
fi

# If port is not provided, derive it from CUDA_VISIBLE_DEVICES:
# - When CUDA_VISIBLE_DEVICES is set, use 9000 + CUDA_VISIBLE_DEVICES
# - Otherwise, default to 9000
if [ -z "$port" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        port=$((9000 + CUDA_VISIBLE_DEVICES))
    else
        port=9000
    fi
    echo "Port not specified, using auto-detected port: $port"
fi

# Rename robot observation keys to policy input keys if necessary. 
# For example, if the robot provides "observation.images.right_camera" but the policy expects "observation.images.right", we can set:
rename_map='{"observation.images.right": "observation.images.right_camera"}'


# Uncomment below if no rename is needed.
rename_map='{}'


python src/tools/policy_server.py \
    --host=0.0.0.0 \
    --port="$port" \
    --policy_type="$policy_type" \
    --pretrained_name_or_path="$ckpt" \
    --policy_device=cuda \
    --rename_map="${rename_map}" \
    --actions_per_chunk=50
