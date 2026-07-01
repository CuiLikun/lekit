#!/bin/bash

dataset="$1"
steps="${2:-30000}"

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset> [steps]"
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python scripts/train_acp.py \
    --job_name="pistar05-${dataset}" \
    --dataset.repo_id="sorel/${dataset}" \
    --dataset.root="data/train_data/lerobot_v3.0/${dataset}" \
    --policy.type="pi05" \
    --policy.pretrained_path="lerobot/pi05_base" \
    --policy.push_to_hub=false \
    --policy.repo_id="sorel/pistar05-${dataset}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --batch_size=16 \
    --steps=${steps} \
    --acp.enable=true \
    --acp.indicator_field="complementary_info.acp_indicator" \
    --acp.indicator_dropout_prob=0.3 \
    --output_dir="outputs/pistar05_policy/${dataset}" \
    --wandb.enable=false
