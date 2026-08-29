#!/bin/bash
# record.sh

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
endpoint="tcp://127.0.0.1:5555"
camera_stream_config="$script_dir/examples/isaac_teleop_to_jaka/camera_stream.yaml"
dataset_name="scoop-powder"
dataset_root="$script_dir/datasets/${dataset_name}_$(date +%Y%m%d_%H%M%S)"
camera_stream_pid=""
recorder_pid=""

camera_stream_ready() {
    uv run camera-stream topic list \
        --endpoint "$endpoint" \
        --timeout 0.5 \
        >/dev/null 2>&1
}

cleanup_camera_stream() {
    if [[ -n "$camera_stream_pid" ]] && kill -0 "$camera_stream_pid" 2>/dev/null; then
        echo "Stopping local camera-stream service..."
        kill -TERM -- "-$camera_stream_pid" 2>/dev/null \
            || kill -TERM "$camera_stream_pid" 2>/dev/null \
            || true
        for _ in {1..30}; do
            kill -0 "$camera_stream_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$camera_stream_pid" 2>/dev/null; then
            echo "camera-stream did not stop after 3 seconds; forcing shutdown." >&2
            kill -KILL -- "-$camera_stream_pid" 2>/dev/null \
                || kill -KILL "$camera_stream_pid" 2>/dev/null \
                || true
        fi
        wait "$camera_stream_pid" 2>/dev/null || true
    fi
}

cleanup_recorder() {
    if [[ -z "$recorder_pid" ]] || ! kill -0 "$recorder_pid" 2>/dev/null; then
        return
    fi

    echo "Stopping recorder..."
    kill -TERM -- "-$recorder_pid" 2>/dev/null \
        || kill -TERM "$recorder_pid" 2>/dev/null \
        || true
    for _ in {1..100}; do
        kill -0 "$recorder_pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$recorder_pid" 2>/dev/null; then
        echo "Recorder did not stop after SIGTERM; forcing shutdown." >&2
        kill -KILL -- "-$recorder_pid" 2>/dev/null \
            || kill -KILL "$recorder_pid" 2>/dev/null \
            || true
    fi
    wait "$recorder_pid" 2>/dev/null || true
    recorder_pid=""
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    cleanup_recorder
    cleanup_camera_stream
    return "$status"
}

start_camera_stream() {
    if camera_stream_ready; then
        echo "Using existing camera-stream service at $endpoint"
        return
    fi

    echo "Starting local camera-stream service..."
    setsid uv run camera-stream server --config "$camera_stream_config" &
    camera_stream_pid=$!
    for _ in {1..40}; do
        if camera_stream_ready; then
            echo "Local camera-stream service is ready."
            return
        fi
        if ! kill -0 "$camera_stream_pid" 2>/dev/null; then
            status=0
            wait "$camera_stream_pid" || status=$?
            echo "camera-stream exited before becoming ready (status $status)." >&2
            ((status != 0)) || status=1
            exit "$status"
        fi
        sleep 0.25
    done
    echo "camera-stream did not become ready within 10 seconds." >&2
    exit 1
}

wait_for_camera_topic() {
    local topic=$1
    echo "Waiting for $topic..."
    if timeout 15 uv run camera-stream topic echo "$topic" \
        --endpoint "$endpoint" \
        --count 1 \
        >/dev/null 2>&1; then
        echo "$topic is ready."
        return
    fi

    echo "$topic did not produce a frame within 15 seconds." >&2
    uv run camera-stream topic info "$topic" \
        --endpoint "$endpoint" \
        --timeout 0.5 \
        >&2 || true
    exit 1
}

[[ -f "$camera_stream_config" ]] || {
    echo "Missing camera-stream configuration: $camera_stream_config" >&2
    exit 1
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_camera_stream
wait_for_camera_topic "base_camera/color"
wait_for_camera_topic "hand_camera/color"

reset_joints="[-0.956, 1.903, 1.427, 1.368, -1.590, -0.290]"

base_camera="base_camera: {type: stream, endpoint: '$endpoint', camera_name: base_camera, width: 640, height: 480, fps: 30}"
hand_camera="hand_camera: {type: stream, endpoint: '$endpoint', camera_name: hand_camera, width: 640, height: 480, fps: 30}"

setsid uv run python -m examples.isaac_teleop_to_jaka.record \
    --robot.type=jaka_robot \
    --robot.id=jaka_arm \
    --robot.ip=192.168.1.31 \
    --robot.cameras="{$base_camera, $hand_camera}" \
    --robot.reset_joints="$reset_joints" \
    --teleop.type=xr_controller \
    --teleop.lock_pose=True \
    --teleop.thumbstick_deadband=0.15 \
    --teleop.thumbstick_angular_speed_rad_s=0.5 \
    --teleop.tool_tip_offset_m="[0.0,0.0,0.27]" \
    --teleop.use_head_yaw=True \
    --teleop.operator_yaw_deg=0 \
    --reset_hold_s=1.0 \
    --dataset.repo_id="sorel/$dataset_name" \
    --dataset.root="$dataset_root" \
    --dataset.single_task="scoop powder from the container" \
    --dataset.fps=30 \
    --dataset.num_episodes=49 \
    --dataset.episode_time_s=9999 \
    --dataset.streaming_encoding=False \
    --dataset.push_to_hub=False \
    --rerun_url="rerun+http://127.0.0.1:9876/proxy" \
    "$@" &
recorder_pid=$!

if [[ -n "$camera_stream_pid" ]]; then
    while kill -0 "$recorder_pid" 2>/dev/null; do
        if ! kill -0 "$camera_stream_pid" 2>/dev/null; then
            camera_status=0
            wait "$camera_stream_pid" || camera_status=$?
            camera_stream_pid=""
            echo "camera-stream exited while recording (status $camera_status); stopping safely." >&2
            cleanup_recorder
            ((camera_status != 0)) || camera_status=1
            exit "$camera_status"
        fi
        sleep 0.1
    done
fi

record_status=0
wait "$recorder_pid" || record_status=$?
recorder_pid=""
exit "$record_status"
