#!/bin/bash

dataset="$1"

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

policy_type="pi05"
rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

# For RTX 4000 series GPUs to avoid NCCL errors
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python src/lerobot/scripts/lerobot_train.py\
  --job_name="train-${policy_type}-${dataset}" \
    --dataset.repo_id="sorel/${dataset}" \
    --policy.repo_id="sorel/${policy_type}-${dataset}" \
    --policy.type=${policy_type} \
    --policy.pretrained_path="lerobot/pi05_base" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --wandb.enable=true \
    --batch_size=32 \
    --steps=50_000
