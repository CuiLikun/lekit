#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)
PIPER_CONFIG="$PROJECT_ROOT/configs/piper_lehub.json"
PIPER_TRANSLATION_SCALE=${LEHUB_PIPER_TRANSLATION_SCALE:-1.0}
PIPER_ROTATION_SCALE=${LEHUB_PIPER_ROTATION_SCALE:-0.1}
HUB_PORT=8080
PIDS=()
NAMES=()
CLEANING_UP=0

usage() {
  cat <<'EOF'
Start LeHub, Quest 3 teleop, and the Piper Robot as one supervised group.

Usage:
  ./src/lekit/scripts/start_lehub.sh

Environment:
  LEHUB_HOST       LAN IPv4 advertised to Quest and browser clients.
                   Defaults to the source address of the default IPv4 route.
  LEHUB_PIPER_ROTATION_SCALE
                   Quest-to-Piper rotation gain. Defaults to 0.1.
  LEHUB_PIPER_TRANSLATION_SCALE
                   Quest-to-Piper translation gain. Defaults to 1.0.
EOF
}

fail() {
  echo "LeHub startup failed: $*" >&2
  exit 1
}

cleanup() {
  if ((CLEANING_UP)); then
    return
  fi
  CLEANING_UP=1
  trap - EXIT INT TERM
  echo
  echo "Stopping LeHub nodes..."
  for ((index = ${#PIDS[@]} - 1; index >= 0; index--)); do
    pid=${PIDS[index]}
    if kill -0 "$pid" 2>/dev/null; then
      echo "  stopping ${NAMES[index]} (pid $pid)"
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for _ in {1..30}; do
    alive=0
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    ((alive == 0)) && break
    sleep 0.1
  done
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
}

start_node() {
  local name=$1
  shift
  echo "Starting $name..."
  setsid "$@" &
  PIDS+=("$!")
  NAMES+=("$name")
  sleep 0.05
}

wait_for_hub() {
  local hub_pid=${PIDS[0]}
  for _ in {1..50}; do
    if curl -fsS --max-time 0.2 "http://127.0.0.1:$HUB_PORT/api/snapshot" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$hub_pid" 2>/dev/null; then
      wait "$hub_pid"
      return $?
    fi
    sleep 0.1
  done
  echo "Hub did not become ready within 5 seconds." >&2
  return 1
}

if (($#)); then
  if (($# == 1)) && [[ $1 == "--help" || $1 == "-h" ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 2
fi

for command in uv ip ss curl setsid; do
  command -v "$command" >/dev/null 2>&1 || fail "required command '$command' was not found"
done
[[ -f $PIPER_CONFIG ]] || fail "missing Piper configuration: $PIPER_CONFIG"
[[ -n $(ip -o link show up dev can0 2>/dev/null) ]] || fail "can0 is missing or not UP"

ADVERTISE_HOST=${LEHUB_HOST:-}
if [[ -z $ADVERTISE_HOST ]]; then
  route=$(ip -4 route get 1.1.1.1 2>/dev/null) || fail "cannot determine the default IPv4 route"
  if [[ $route =~ [[:space:]]src[[:space:]]([^[:space:]]+) ]]; then
    ADVERTISE_HOST=${BASH_REMATCH[1]}
  else
    fail "cannot determine a LAN IPv4; set LEHUB_HOST explicitly"
  fi
fi
[[ $ADVERTISE_HOST != *[[:space:]/:]* ]] || fail "LEHUB_HOST must be an IPv4 address without a port"

for port in 5557 5560 8000 8080 8081; do
  [[ -z $(ss -H -ltn "sport = :$port" 2>/dev/null) ]] || fail "TCP port $port is already in use"
done

cd "$PROJECT_ROOT" || fail "cannot enter project root"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "LeHub host: $ADVERTISE_HOST"
start_node Hub uv run lekit hub \
  --management-endpoint tcp://0.0.0.0:5560 \
  --advertise-host "$ADVERTISE_HOST" \
  --web-host 0.0.0.0 \
  --web-port "$HUB_PORT" \
  --auto-route-single-pair

wait_for_hub || exit $?

start_node "Quest 3 teleop" uv run lekit teleop \
  --hub-seed "tcp://$ADVERTISE_HOST:5560" \
  --action-endpoint tcp://0.0.0.0:5557 \
  --monitor-host 0.0.0.0 \
  --monitor-port 8000 \
  --advertise-host "$ADVERTISE_HOST"

start_node "Piper Robot" uv run lekit robot \
  --kind piper \
  --hub-seed "tcp://$ADVERTISE_HOST:5560" \
  --robot-config "$PIPER_CONFIG" \
  --video-host 0.0.0.0 \
  --video-port 8081 \
  --advertise-host "$ADVERTISE_HOST" \
  --control-rate-hz 30 \
  --processor.translation_scale "$PIPER_TRANSLATION_SCALE" \
  --processor.rotation_scale "$PIPER_ROTATION_SCALE" \
  --processor.max_translation_from_anchor_m 0.10 \
  --processor.max_rotation_from_anchor_rad 0.17453292519943295 \
  --enable-motion

echo
echo "LeHub:         http://$ADVERTISE_HOST:8080"
echo "Quest monitor: http://$ADVERTISE_HOST:8000"
echo "Robot cameras: http://$ADVERTISE_HOST:8081/api/cameras"
echo "Press Ctrl-C once to stop all nodes."

wait -n "${PIDS[@]}"
status=$?
echo "A node exited with status $status; stopping the group." >&2
exit "$status"
