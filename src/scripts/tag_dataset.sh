#!/usr/bin/env bash
# Tag a dataset on Hugging Face Hub.
# Usage: ./tag_dataset.sh <repo_id> <tag> <repo_type>

set -euo pipefail

REPO_ID="${1:-sorel/so101-record-0121}"
TAG="${2:-v3.0}"
REPO_TYPE="dataset"

if [ -z "$REPO_ID" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 <repo_id> <tag> <repo_type>"
    exit 1
fi

python - "$REPO_ID" "$TAG" "$REPO_TYPE" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, tag, repo_type = sys.argv[1], sys.argv[2], sys.argv[3]
HfApi().create_tag(repo_id, tag=tag, repo_type=repo_type)
print(f"Created tag '{tag}' for {repo_type} repo '{repo_id}'.")
PY
