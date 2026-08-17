from dataclasses import dataclass
from pathlib import Path

import torch
from rich import print

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


@dataclass
class ReplayBotConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Index of the episode(s) to replay. Can be a single episode index or a comma-separated list of indices (e.g. "0,1,2").
    episodes: str | None = None  # If None, replay all episodes in the dataset.
    # Robot type
    type: str = "replay_bot"


class ReplayBot:
    """A lightweight dataset-backed robot for deterministic episode replay.

    Only two pieces of mutable runtime state are tracked: the currently loaded
    episode dataset and the current frame index. They are guarded by a re-entrant
    lock so `capture_observation()` can safely be called from a background thread.
    """

    name = "replay_bot"

    def __init__(self, config: ReplayBotConfig):
        self.config = config
        self.ds_meta = LeRobotDatasetMetadata(config.repo_id, root=config.root)
        self.camera_keys = tuple(self.ds_meta.camera_keys)
        self.total_episodes = self.ds_meta.total_episodes
        self.episodes = self._parse_episodes(config.episodes, self.ds_meta.total_episodes)

        self.dataset = None  # Current episode dataset
        self.frame_index = 0  # Current frame index within the loaded episode
        self.frame_cache = None

        print(f"Total episodes in dataset {config.repo_id}: {self.ds_meta.total_episodes}")
        print(f"Camera keys: {self.camera_keys}")
        print(f"Feature keys: {self.ds_meta.features.keys()}")

    @staticmethod
    def _parse_episodes(episode_spec: str | None, total_episodes: int) -> list[int]:
        if not episode_spec:
            return list(range(total_episodes))

        episodes = [int(ep.strip()) for ep in episode_spec.split(",") if ep.strip()]
        if not episodes:
            raise ValueError("`episode` must contain at least one valid episode index.")
        return episodes

    def load_episode(self, episode_index: int):
        print(f"Loading episode {episode_index} from dataset {self.config.repo_id}...")
        self.dataset = LeRobotDataset(self.config.repo_id, root=self.config.root, episodes=[episode_index])
        self.frame_index = 0
        self.frame_cache = None

    def step(self):
        """Advance to the next frame in the currently loaded episode."""
        dataset = self.dataset
        if dataset is not None and self.frame_index < dataset.num_frames:
            self.frame_index += 1
        self.frame_cache = None

    @property
    def is_episode_done(self) -> bool:
        return self.dataset is not None and self.frame_index >= self.dataset.num_frames

    @property
    def teleop_action(self) -> torch.Tensor:
        """Return the recorded action for the current frame."""
        if self.frame_cache is None:
            self.frame_cache = self.dataset[self.frame_index]
        return self.frame_cache["action"]

    def capture_observation(self) -> dict[str, torch.Tensor | str] | None:
        """Return a thread-safe snapshot of the current observation."""
        if self.frame_cache is None:
            self.frame_cache = self.dataset[self.frame_index]
        return {
            "task": self.frame_cache["task"],
            "observation.state": self.frame_cache["observation.state"],
            **{key: self.frame_cache[key] for key in self.camera_keys},
        }

    def get_observation(self) -> dict[str, torch.Tensor | str] | None:
        """Backward-compatible alias used by some replay utilities."""
        return self.capture_observation()

    def get_teleop_action(self) -> dict[str, torch.Tensor]:
        """Return the recorded action in the same shape as live robot APIs."""
        return {"action": self.teleop_action}

    def send_action(self, action: torch.Tensor):
        """Replay mode does not actuate hardware; keep the interface consistent."""
        return {"action": action}

    def get_state_value(self) -> torch.Tensor:
        """Mock state value inference for demonstration purposes."""
        if self.frame_cache is None:
            self.frame_cache = self.dataset[self.frame_index]
        return self.frame_cache.get("complementary_info.value", 0.0)
