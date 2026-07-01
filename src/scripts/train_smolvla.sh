#!/bin/bash

dataset="$1"
steps="${2:-50_000}"

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --job_name="smolvla-${dataset}" \
  --dataset.repo_id="sorel/${dataset}" \
  --policy.repo_id="sorel/smolvla-${dataset}" \
  --policy.path="lerobot/smolvla_base" \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --wandb.enable=true \
  --batch_size=32 \
  --steps=${steps}
