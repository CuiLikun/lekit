# Merge train and validation splits back into one dataset


lerobot-edit-dataset \
    --repo_id Sorel/dataset-merged \
    --operation.type merge \
    --operation.repo_ids "['sorel/record-0121', 'sorel/record-0121-p1', 'sorel/record-0122', 'sorel/record-0123']"