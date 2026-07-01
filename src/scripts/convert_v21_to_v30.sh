#!/bin/bash

python ./src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
 --repo-id="record_1212" \
 --root="/root/lerobot/tmp" \
 --push-to-hub=False
