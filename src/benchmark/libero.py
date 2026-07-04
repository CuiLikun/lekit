import time
from pathlib import Path

from lerobot.datasets import LeRobotDataset

from utils.rerun_utils import RerunLogger

if __name__ == "__main__":
    # Load the dataset
    local_root = Path("./data/libero")
    dataset = LeRobotDataset("lerobot/libero", root=local_root if local_root.exists() else None)

    # Print dataset statistics
    print(dataset)

    # Use the RerunLogger to visualize the dataset
    rerun = RerunLogger()
    ep_id = -1
    for frame in dataset:
        if frame["episode_index"] != ep_id:
            rerun.switch_record()
            ep_id = frame["episode_index"]
            print(f"Switch to episode {ep_id}")
        rerun.log(frame)
        time.sleep(0.033)  # Simulate real-time playback at ~30 FPS
