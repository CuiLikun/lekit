#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

dataset="$1"
ckpt="$2"

if [ -z "$dataset" ] || [ -z "$ckpt" ]; then
    echo "Usage: $0 <dataset> <ckpt>"
    exit 1
fi

# where the training data is located
data_dir="data/train_data/lerobot_v3.0"

python scripts/infer_value.py \
    --job_name="infer-value-${dataset}" \
    --dataset.repo_id="sorel/${dataset}" \
    --dataset.root="${data_dir}/${dataset}" \
    --inference.checkpoint_path="${ckpt}" \
    --runtime.device=cuda \
    --runtime.batch_size=64 \
    --acp.enable=true \
    --acp.n_step=50 \
    --acp.positive_ratio=0.3 \
    --acp.value_field="complementary_info.value" \
    --acp.advantage_field="complementary_info.advantage" \
    --acp.indicator_field="complementary_info.acp_indicator" \
    --viz.enable=false \
    --output_dir="outputs/value_infer/value-${dataset}"
