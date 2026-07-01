#!/bin/bash

dataset="record_0429"

# # Label specified episodes as failure, and the rest as success
# python scripts/tools/label_outcomes.py \
#     --repo-id="sorel/${dataset}" \
#     --root="data/train_data/lerobot_v3.0/${dataset}" \
#     --failure-episodes="0,2,5"

# # Label specified episodes as success, and the rest as failure
# python scripts/tools/label_outcomes.py \
#     --repo-id="sorel/${dataset}" \
#     --root="data/train_data/lerobot_v3.0/${dataset}" \
#     --success-episodes="0,2,5"

# Label all episodes as success
python scripts/tools/label_outcomes.py \
    --repo-id="sorel/${dataset}" \
    --root="data/train_data/lerobot_v3.0/${dataset}" \
    --all-outcome="success"

# # Label all episodes as failure
# python scripts/tools/label_outcomes.py \
#     --repo-id="sorel/${dataset}" \
#     --root="data/train_data/lerobot_v3.0/${dataset}" \
#     --all-outcome="failure"
