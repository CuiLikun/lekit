"""Replay a LeRobotDataset back into a SimRobot.

Demonstrates that an action sequence recorded by SimRobot can be replayed
onto a SimRobot and produce sensible state trajectories -- the round trip
the LeRobot pipeline promises.

Usage::

    MUJOCO_GL=egl PYTHONPATH=src python examples/sim_robot/replay.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Make ``from hardwares import …`` resolve when this script is launched with
# PYTHONPATH=src. Also keep HF datasets cache off the user's read-only $HOME
# (the sandbox here has ``~/.cache/huggingface`` mounted read-only).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
_TMP_HOME = Path(tempfile.gettempdir()) / "sim_robot_hf"
_TMP_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_TMP_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(_TMP_HOME / "datasets"))

import numpy as np  # noqa: E402

from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

from hardwares import (  # noqa: E402
    SimCameraConfig,
    SimMotorSpec,
    SimRobot,
    SimRobotConfig,
)


def _say(msg: str) -> None:
    """One-line progress reporter. Plain ``print`` to stderr to avoid the
    post-MuJoCo-OpenGL-init console black-hole that suppresses ``logging``."""
    print(msg, file=sys.stderr, flush=True)


def build_robot(scene: Path, calibration_dir: Path) -> SimRobot:
    """Same robot configuration as ``record.py``."""
    cfg = SimRobotConfig(
        id="replay_example",
        xml_path=str(scene),
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
        max_relative_target=0.5,  # let the PD controller track the recorded trajectory
        n_substeps=4,
        calibration_dir=calibration_dir,
    )
    return SimRobot(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path(__file__).parent / "scene.xml")
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/sim_robot_dataset"))
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Subset of episode indices to replay (default: all).")
    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}. Run record.py first.")

    dataset = LeRobotDataset("sim_robot_demo", root=args.dataset_root, episodes=args.episodes)
    _say(f"[replay] dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames, "
         f"features={list(dataset.features)}")

    action_names = dataset.features[ACTION]["names"]
    if action_names != [f"{m}.pos" for m in ["shoulder", "elbow", "wrist"]]:
        raise SystemExit(
            f"Unexpected action names in dataset: {action_names}. "
            "This example assumes the layout recorded by record.py."
        )

    # ``select_columns`` returns a ``datasets.Dataset`` of just the action
    # vectors so we never decode images we don't need.
    actions_only = dataset.select_columns(ACTION)
    action_vec_iter = (np.asarray(row[ACTION], dtype=np.float32) for row in actions_only)

    robot = build_robot(args.scene, args.dataset_root / "calibration")
    robot.connect()
    try:
        episode_idx = 0
        frame_in_episode = 0
        diffs_per_episode: list[np.ndarray] = []
        per_episode_summary: list[dict] = []

        def _flush_episode(curr_episode_idx: int, curr_frame_count: int) -> None:
            if not diffs_per_episode:
                return
            arr = np.stack(diffs_per_episode)
            summary = {
                "episode": curr_episode_idx,
                "frames": curr_frame_count,
                "max_abs_delta": float(arr.max()),
                "mean_abs_delta": float(arr.mean()),
            }
            per_episode_summary.append(summary)
            _say(f"[replay] === episode {summary['episode']} done ({summary['frames']} frames) ===")
            _say(f"[replay]   max per-step state delta: {summary['max_abs_delta']:.4f} rad")
            _say(f"[replay]   mean per-step state delta: {summary['mean_abs_delta']:.4f} rad")

        for frame_idx in range(dataset.num_frames):
            raw = dataset.hf_dataset[frame_idx]
            current_episode = int(raw["episode_index"])
            if current_episode != episode_idx:
                _flush_episode(episode_idx, frame_in_episode)
                episode_idx = current_episode
                frame_in_episode = 0
                diffs_per_episode = []
                robot.configure()  # rewind sim to its initial pose

            frame_in_episode += 1
            target_vec = next(action_vec_iter)
            target = dict(zip(action_names, target_vec.tolist()))

            robot.send_action(target)
            obs = robot.get_observation()
            actual_vec = np.array([obs[n] for n in action_names], dtype=np.float32)
            diffs_per_episode.append(np.abs(actual_vec - target_vec))

        # Report the final episode.
        _flush_episode(episode_idx, frame_in_episode)

        if per_episode_summary:
            overall_max = max(s["max_abs_delta"] for s in per_episode_summary)
            overall_mean = np.mean([s["mean_abs_delta"] for s in per_episode_summary])
            _say(f"[replay] summary over {len(per_episode_summary)} episodes: "
                 f"overall max delta = {overall_max:.4f} rad, overall mean = {overall_mean:.4f} rad")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
