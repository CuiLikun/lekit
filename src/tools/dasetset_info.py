#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load a LeRobot dataset and print its summary information."""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class DatasetInfoConfig:
    # Dataset repo id, for example: lerobot/pusht or record_0330.
    repo_id: str
    # Optional local dataset root path.
    root: str | Path | None = None
    # Print the full feature schema.
    show_features: bool = False
    # Print task definitions.
    show_tasks: bool = False


def get_dataset_size_bytes(repo_path: str | Path) -> int:
    """Recursively compute the total size of a local dataset directory."""
    total = 0
    with os.scandir(repo_path) as entries:
        for entry in entries:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dataset_size_bytes(entry.path)
    return total


def format_size(size_bytes: int) -> str:
    """Format a byte count using binary units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def print_json_block(title: str, payload: Any) -> None:
    """Print a JSON-formatted block."""
    print(f"{title}:")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@parser.wrap()
def main(cfg: DatasetInfoConfig) -> None:
    """Load the dataset and print a compact summary."""
    root = str(cfg.root) if cfg.root is not None else None

    dataset = LeRobotDataset(cfg.repo_id, root=root)
    average_frames_per_episode = dataset.meta.total_frames / dataset.meta.total_episodes
    average_episode_seconds = average_frames_per_episode / dataset.meta.fps

    print(f"====== Dataset Info: {dataset.meta.repo_id}")
    print(f"Repository ID: {dataset.meta.repo_id}")
    print(f"Root: {dataset.root}")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"Total tasks: {dataset.meta.total_tasks}")
    print(f"Total frames: {dataset.meta.total_frames}")
    print(f"Loaded frames: {len(dataset)}")
    print(f"FPS: {dataset.meta.fps}")
    print(f"Robot type: {dataset.meta.robot_type}")
    print(f"Camera keys: {list(dataset.meta.camera_keys)}")
    print(f"Average frames per episode: {average_frames_per_episode:.1f}")
    print(f"Average episode length: {average_episode_seconds:.1f} s")

    dataset_root = Path(dataset.root)
    if dataset_root.exists():
        print(f"Local size: {format_size(get_dataset_size_bytes(dataset_root))}")

    if cfg.show_tasks:
        print_json_block("Tasks", dataset.meta.tasks)

    if cfg.show_features:
        print_json_block("Features", dataset.meta.features)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
