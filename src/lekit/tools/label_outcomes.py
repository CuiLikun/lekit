#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Label episode outcomes and write them into episodes metadata.

Examples:
    Mark all episodes as success:
        python scripts/tools/label_outcomes.py --repo-id lerobot/pusht --all-outcome success

    Mark all episodes as failure:
        python scripts/tools/label_outcomes.py --repo-id lerobot/pusht --all-outcome failure

    Mark selected episodes as success, all others as failure:
        python scripts/tools/label_outcomes.py --repo-id lerobot/pusht --success-episodes 0,2,5

    Mark selected episodes as failure:
        python scripts/tools/label_outcomes.py --repo-id lerobot/pusht --failure-episodes 0,2,5
"""

import argparse

from lerobot.datasets.dataset_tools import add_episode_field
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _parse_episode_list(episode_text: str, arg_name: str) -> set[int]:
    """Parse comma-separated episode indices into a set."""
    if not episode_text.strip():
        return set()

    try:
        return {int(ep.strip()) for ep in episode_text.split(",") if ep.strip()}
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Label episode outcomes in LeRobot dataset metadata")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id, e.g. lerobot/pusht")
    parser.add_argument("--root", type=str, default=None, help="Optional local dataset root")

    parser.add_argument(
        "--success-episodes",
        type=str,
        default="",
        help="Comma-separated successful episode indices (e.g. 0,1,2)",
    )
    parser.add_argument(
        "--failure-episodes",
        type=str,
        default="",
        help="Comma-separated failed episode indices (e.g. 0,1,2)",
    )
    parser.add_argument(
        "--all-outcome",
        type=str,
        choices=("success", "failure"),
        default=None,
        help="Mark every episode with the same outcome",
    )

    args = parser.parse_args()

    if args.all_outcome is not None and (args.success_episodes or args.failure_episodes):
        parser.error("Use --all-outcome by itself, or one of --success-episodes/--failure-episodes")

    if args.success_episodes and args.failure_episodes:
        parser.error("Use only one of --success-episodes or --failure-episodes")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    if args.all_outcome is not None:
        values = dict.fromkeys(range(dataset.meta.total_episodes), args.all_outcome)
        add_episode_field(
            dataset=dataset,
            field_name="outcome",
            values=values,
        )

        print(
            f"Updated 'outcome' for {dataset.meta.total_episodes} episodes in {dataset.repo_id} at {dataset.root}"
        )
        return

    success_episodes = _parse_episode_list(args.success_episodes, "--success-episodes")
    failure_episodes = _parse_episode_list(args.failure_episodes, "--failure-episodes")

    chosen_episodes = success_episodes if success_episodes else failure_episodes
    invalid = chosen_episodes - set(range(dataset.meta.total_episodes))
    if invalid:
        bad_arg = "--success-episodes" if success_episodes else "--failure-episodes"
        parser.error(f"Invalid episode indices in {bad_arg}: {sorted(invalid)}")

    if success_episodes:
        values = {
            ep_idx: ("success" if ep_idx in success_episodes else "failure")
            for ep_idx in range(dataset.meta.total_episodes)
        }
    elif failure_episodes:
        values = {
            ep_idx: ("failure" if ep_idx in failure_episodes else "success")
            for ep_idx in range(dataset.meta.total_episodes)
        }
    else:
        values = dict.fromkeys(range(dataset.meta.total_episodes), "failure")

    add_episode_field(
        dataset=dataset,
        field_name="outcome",
        values=values,
    )

    print(
        f"Updated 'outcome' for {dataset.meta.total_episodes} episodes in {dataset.repo_id} at {dataset.root}"
    )


if __name__ == "__main__":
    main()
