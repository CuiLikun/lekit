#!/bin/bash

python -m lerobot.async_inference.robot_client_zmq \
    --host="localhost" \
    --port=8000 \
    --fps=30 \
    --aggregate_fn_name="weighted_average"