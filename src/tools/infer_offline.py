"""
Offline inference script for running a policy on recorded episodes.

This script replays episodes from a dataset while collecting actions from a policy model
and comparing them with the original teleop actions. It supports:
- Real-time inference using PolicyClient
- Episode replay with MockRobot
- Frame-by-frame timing and performance metrics
- Action comparison and visualization
- Optional remote Rerun logging for visualization

Example usage:
python scripts/tools/infer_offline.py \
    --policy.host 127.0.0.1 \
    --policy.port 8001 \
    --robot.repo_id record_0429 \
    --robot.root data/train_data/lerobot_v3.0/record_0429 \
    --robot.episodes 0,1,2 \
    --rerun_url rerun+http://172.20.76.73:9876/proxy
"""

import pickle  # nosec
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
import numpy as np
import torch
import zmq
from compare_actions import compare_actions
from policy_client import PolicyClient, PolicyClientConfig
from rich import print
from rich.progress import Progress

from lerobot.utils.replay_bot import ReplayBot, ReplayBotConfig
from lerobot.utils.rerun_utils import RerunLogger


@dataclass
class InferOfflineConfig:
    # Policy configuration
    policy: PolicyClientConfig
    # Robot configuration
    robot: ReplayBotConfig
    # Limit the frames per second. By default, uses the dataset fps.
    fps: int | None = None
    # Directory to save action data for comparison
    save_dir: str = "outputs/infer_offline"
    # Optional: Remote Rerun viewer URL. If None, use local viewer.
    rerun_url: str | None = None
    # Optional: Value server host. If None, value inference is disabled.
    value_host: str | None = None
    # Value server port (used only when value_host is not None).
    value_port: int = 8000


@dataclass
class ValueClientConfig:
    host: str = "localhost"
    port: int = 8002


class ValueClient:
    def __init__(self, config: ValueClientConfig):
        self.config = config
        self.context = zmq.Context()
        self._socket = self.context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self.config.host}:{self.config.port}")

    def stop(self) -> None:
        self._socket.close(linger=0)
        self.context.term()

    def _request(self, payload: dict) -> dict:
        self._socket.send(pickle.dumps(payload))
        message = pickle.loads(self._socket.recv())
        if not isinstance(message, dict):
            raise RuntimeError(f"Invalid response type from value server: {type(message).__name__}")
        if message.get("error"):
            raise RuntimeError(f"Value server error: {message['error']}")
        return message

    def require_value(self, observation: dict) -> float:
        response = self._request({"observation": observation})
        value = response.get("value")
        if value is None:
            raise RuntimeError("Value server response missing 'value'")

        if isinstance(value, torch.Tensor):
            return float(value.reshape(-1)[0].item())
        return float(np.asarray(value).reshape(-1)[0])


@draccus.wrap()
def infer_offline(config: InferOfflineConfig):
    print(asdict(config))

    robot = ReplayBot(config.robot)
    policy = PolicyClient(config.policy)
    value_client = (
        ValueClient(ValueClientConfig(host=config.value_host, port=config.value_port))
        if config.value_host is not None
        else None
    )

    # Create output directory if it doesn't exist
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    repo_name = config.robot.repo_id.replace("/", "_")

    logger = RerunLogger(url=config.rerun_url) if config.rerun_url else None

    try:
        for episode in robot.episodes:
            robot.load_episode(episode)
            fps = config.fps or robot.ds_meta.info["fps"]

            # Reset policy and replay episode
            policy.reset()

            policy_actions = []
            teleop_actions = []
            predicted_values = []
            dataset_values = []

            if logger:
                logger.switch_record()
            # Replay episode until done with MockRobot providing observations and PolicyClient providing actions
            with Progress() as progress:
                task_id = progress.add_task("Replaying episode", total=robot.dataset.num_frames)
                while not robot.is_episode_done:
                    start = time.perf_counter()

                    # Get action from policy
                    t0 = time.perf_counter()
                    observation = robot.capture_observation()
                    obs_time = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    policy_action = policy.require_action(observation).numpy()
                    policy_time = time.perf_counter() - t0

                    value_time = 0.0
                    predicted_value = None
                    if value_client is not None:
                        t0 = time.perf_counter()
                        predicted_value = value_client.require_value(observation)
                        value_time = time.perf_counter() - t0

                    teleop_action = robot.get_teleop_action()["action"].numpy()

                    # Store actions for later comparison
                    teleop_actions.append(teleop_action)
                    policy_actions.append(policy_action)

                    # Log to Rerun
                    if logger is not None:
                        data = {
                            **observation,
                            "teleop": teleop_action,
                            "policy": policy_action,
                            "state_value": predicted_value + 1,
                            "framestep": len(policy_actions) - 1,
                        }
                        logger.log(data)

                    # Send action to the robot (which will advance to the next frame)
                    robot.send_action(action=policy_action)
                    robot.step()  # Advance to the next frame

                    # Busy-wait to maintain the desired fps
                    time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - start)))

                    frame_interval = time.perf_counter() - start
                    real_fps = 1.0 / frame_interval if frame_interval > 0 else float("inf")

                    progress.update(
                        task_id,
                        advance=1,
                        description=(
                            f"Replaying {len(policy_actions)}/{robot.dataset.num_frames} "
                            f"(frame={frame_interval * 1000:.2f}ms, fps={real_fps:.2f}, "
                            f"obs={obs_time * 1000:.2f}ms, policy={policy_time * 1000:.2f}ms, "
                            f"value={value_time * 1000:.2f}ms, "
                        ),
                    )

            policy_actions = np.array(policy_actions)
            teleop_actions = np.array(teleop_actions)
            predicted_values = np.array(predicted_values, dtype=np.float32)
            dataset_values = np.array(dataset_values, dtype=np.float32)
            print(f"{teleop_actions.shape=}, {policy_actions.shape=}, {robot.dataset.num_frames=}")

            # Save action data for later comparison
            np.save(save_path / f"teleop_actions_{repo_name}_episode_{episode}_tel.npy", teleop_actions)
            np.save(
                save_path / f"policy_actions_{repo_name}_episode_{episode}_{policy.policy_name}.npy",
                policy_actions,
            )

            # Generate comparison plot right after each episode replay completes.
            compare_actions(
                data_dir=config.save_dir,
                repo_id=config.robot.repo_id,
                episode=episode,
                policy_types=[policy.policy_name],
                output_dir=config.save_dir,
            )

    finally:
        if logger:
            logger.stop()
        if value_client:
            value_client.stop()
        policy.stop()


if __name__ == "__main__":
    infer_offline()
