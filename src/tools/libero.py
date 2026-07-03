import time

from lerobot.datasets import LeRobotDataset

from utils.rerun_utils import RerunLogger

if __name__ == "__main__":
    # Load the dataset
    dataset = LeRobotDataset("lerobot/libero", root="./data/libero")

    # Print dataset statistics
    print(dataset)

    # Use the RerunLogger to visualize the dataset
    rerun = RerunLogger()
    for frame in dataset:
        rerun.log(frame)
        time.sleep(0.033)  # Simulate real-time playback at ~30 FPS
