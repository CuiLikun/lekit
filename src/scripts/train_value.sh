#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

dataset="$1"
steps="${2:-10_000}"

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset> [steps]"
    exit 1
fi

# where the training data is located
data_dir="data/train_data/lerobot_v3.0"
# where to save checkpoints and logs during training
ckpt_dir="outputs/value_train"

# print dataset report for debugging
# lerobot-dataset-report \
#   --dataset="${dataset}" \
#   --root="${data_dir}"

# train the value function
python scripts/train_value.py \
  --job_name="value-${dataset}" \
  --dataset.repo_id="sorel/${dataset}" \
  --dataset.root="${data_dir}/${dataset}" \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.push_to_hub=false \
  --value.repo_id="sorel/${dataset}-value" \
  --batch_size=16 \
  --steps="${steps}" \
  --output_dir="${ckpt_dir}/${dataset}" \
  --wandb.enable=true