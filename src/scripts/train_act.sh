#!/bin/bash

dataset="$1"
steps="${2:-50_000}"

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --job_name="act-${dataset}" \
  --dataset.repo_id="sorel/${dataset}" \
  --dataset.root="data/train_data/lerobot_v3.0/${dataset}" \
  --policy.type="act" \
  --policy.repo_id="sorel/act-${dataset}" \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=8 \
  --steps=${steps}
