"""Record scripted-rollout episodes from ``SimRobot`` into a LeRobotDataset.

This is the SimRobot counterpart to ``examples/phone_to_so100/record.py`` --
it uses the simulated arm in place of the physical follower and a scripted
sine trajectory in place of the human teleoperator. Everything else (the
``LeRobotDataset`` writer, episode-buffered recording, parquet+videos on
disk) works exactly the same.

Usage::

    MUJOCO_GL=egl PYTHONPATH=src python examples/sim_robot/record.py
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import hw_to_dataset_features

# Make ``from hardwares import …`` resolve when this file is launched as a
# script with PYTHONPATH=src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hardwares import SimCameraConfig, SimMotorSpec, SimRobot, SimRobotConfig  # noqa: E402

logger = logging.getLogger("examples.sim_robot.record")


def build_robot(scene: Path) -> SimRobot:
    """Build a ``SimRobot`` matching the SO101-like demo scene."""
    cfg = SimRobotConfig(
        id="example",
        xml_path=str(scene),
        # motors ←→ MuJoCo joints/actuators defined in scene.xml
        motors={
            "shoulder": SimMotorSpec(joint="shoulder", actuator="motor_shoulder"),
            "elbow":    SimMotorSpec(joint="elbow",    actuator="motor_elbow"),
            "wrist":    SimMotorSpec(joint="wrist",    actuator="motor_wrist"),
        },
        cameras={
            "overhead": SimCameraConfig(name="overhead",  width=128, height=96, fps=30),
            "wrist":    SimCameraConfig(name="wrist_cam", width=96,  height=96, fps=30),
        },
        default_joint_positions={"shoulder": 0.0, "elbow": 0.4, "wrist": -0.6},
        max_relative_target=0.40,  # tuned for stable kp/kv=40/12
        n_substeps=4,
    )
    return SimRobot(cfg)


def scripted_action(t: float, episode: int) -> dict[str, float]:
    """Smooth per-step target for the three motors.

    Different per-episode phase offsets ensure each episode covers a distinct
    trajectory (useful for inspecting replay diversity).
    """
    phase = 0.6 * episode
    return {
        "shoulder.pos": 0.9 * math.sin(0.7 * t + phase),
        "elbow.pos":    0.5 + 0.4 * math.cos(0.5 * t + phase),
        "wrist.pos":   -0.3 * math.sin(1.3 * t + 0.3 * phase),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path(__file__).parent / "scene.xml")
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/sim_robot_dataset"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Clean any prior dataset so we always start fresh.
    if args.dataset_root.exists():
        shutil.rmtree(args.dataset_root)

    robot = build_robot(args.scene)

    # Build the LeRobot feature schema by translating the robot's hardware
    # features -- exactly the same code path ``lerobot.record`` uses for a
    # physical arm.
    features: dict = {}
    features.update(hw_to_dataset_features(robot.action_features, prefix=ACTION, use_video=True))
    features.update(hw_to_dataset_features(robot.observation_features, prefix="observation", use_video=True))

    dataset = LeRobotDataset.create(
        repo_id="sim_robot_demo",
        fps=args.fps,
        robot_type=robot.name,
        root=args.dataset_root,
        features=features,
        use_videos=True,
        image_writer_threads=2,
    )

    robot.connect()
    try:
        joint_keys = [f"{m}.pos" for m in robot.config.motors]
        cam_keys = list(robot.config.cameras)

        for ep in range(args.episodes):
            logger.info("=== Recording episode %d/%d ===", ep + 1, args.episodes)
            # ``configure`` rewinds the simulator to its initial pose so each
            # episode starts identically.
            robot.configure()

            t0 = time.perf_counter()
            for frame_idx in range(args.frames_per_episode):
                # 1. Action first so we capture the observation that follows
                #    the step (closer to how real robot pipelines record).
                target = scripted_action(t=frame_idx / args.fps, episode=ep)
                # SimRobot.send_action takes a dict keyed by ``"motor.pos"``.
                sent_action = robot.send_action(target)

                # 2. Read the post-step observation.
                obs = robot.get_observation()
                state = np.array([obs[k] for k in joint_keys], dtype=np.float32)

                # 3. Hand the frame to the dataset writer.
                frame = {
                    "observation.state": state,
                    ACTION: np.array([sent_action[k] for k in joint_keys], dtype=np.float32),
                    "task": "sim_robot demo",  # plain string per LeRobot schema
                }
                for cam in cam_keys:
                    # Convert (H, W, 3) uint8 RGB to (3, H, W) for LeRobot's
                    # channel-first convention.
                    img = obs[cam]
                    frame[f"observation.images.{cam}"] = img  # np.uint8 (H, W, 3) as LeRobot expects

                dataset.add_frame(frame)

                # Light pacing so the simulator doesn't outrun any downstream
                # video encoder.
                target_dt = 1.0 / args.fps
                elapsed = time.perf_counter() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                t0 = time.perf_counter()

            dataset.save_episode()
            logger.info("  saved %d frames", args.frames_per_episode)

    finally:
        robot.disconnect()
        dataset.finalize()

    # Quick post-mortem so the user can see the size of the dataset.
    info = []
    for cam in ["overhead", "wrist"]:
        info.append(f"images.{cam}={cam in robot.config.cameras}")
    logger.info("Done. Dataset written to %s", args.dataset_root)
    logger.info("  features: %s", sorted(dataset.features))


if __name__ == "__main__":
    main()
