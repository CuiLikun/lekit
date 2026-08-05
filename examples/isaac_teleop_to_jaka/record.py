#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Record a LeRobot dataset by driving a JAKA arm with an Isaac Teleop XR controller.

The XR controller is clutch-rebased onto the measured JAKA TCP pose. Each engaged frame
becomes an absolute ``ee.x/y/z/roll/pitch/yaw`` target (metres and radians) sent through
the JAKA SDK ``servo_p`` Cartesian Servo Move. No host-side inverse kinematics.

Usage:

    python -m examples.isaac_teleop_to_jaka.record \
        --robot.type=jaka_robot \
        --robot.ip=192.168.1.31 \
        --robot.id=jaka_arm \
        --robot.control_mode=ee_pose \
        --robot.servo_step_num=4 \
        --teleop.type=xr_controller \
        --robot.cameras="{ hand: {type: intelrealsense, serial_number_or_name: '342522070741', width: 640, height: 480, fps: 30}}" \
        --dataset.repo_id=<hf_user>/<dataset_name> \
        --dataset.single_task="Pick up the object" \
        --dataset.fps=30 \
        --dataset.num_episodes=3 \
        --dataset.episode_time_s=20 \
        --dataset.reset_time_s=5

Keyboard shortcuts: Right/n ends and saves the current episode, Left/r discards and
re-records it, and Esc/q stops after the current episode. All frames, including
clutch-disengaged hold frames, are recorded.
"""

import logging
import math
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers "opencv")
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401  (registers "intelrealsense")
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from src.robots.jaka_robot import JakaRobot, JakaRobotConfig

from .xr import CLOUDXR_ENV_FILE, IsaacTeleopConfig, make_xr_device

# ── Hold latch ──────────────────────────────────────────────────────────────


class HoldLatch:
    """Re-send the measured action while the device is idle.

    Latching (instead of re-reading each idle frame) prevents a steady-state
    servo error from compounding downward under gravity: a fresh re-command of
    the measurement would lower the goal by that error on every frame.
    """

    def __init__(self, action_keys: list[str]):
        self._action_keys = action_keys
        self._held: dict[str, float] | None = None

    def resolve(self, action: dict | None, obs: dict) -> dict:
        if action is not None:
            self._held = None
            return action
        if self._held is None:
            self._held = {k: float(obs[k]) for k in self._action_keys if k in obs}
        return self._held


# ── Keyboard control ────────────────────────────────────────────────────────


def init_keyboard_listener():
    """Wire Right/Left/Esc shortcuts to recording control events.

    When stdin is a TTY, prefer the stdlib :class:`TerminalKeyListener` (works over SSH
    and emits canonical key names); the dispatcher maps ``n``/``r``/``q`` to the same
    events as the arrow keys / ``Esc`` so a laggy terminal splitting escape sequences
    still gets through. Otherwise fall back to the upstream pynput-based listener.
    """
    if not (sys.stdin is not None and sys.stdin.isatty()):
        from lerobot.utils.keyboard_input import init_keyboard_listener as _upstream

        return _upstream()

    from lerobot.utils.keyboard_input import TerminalKeyListener, apply_recording_control

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)

    listener = TerminalKeyListener(on_key)
    listener.start()
    logging.info(
        "Keyboard control via terminal — keep this terminal focused: "
        "Right/n = end episode early, Left/r = re-record, Esc/q = stop."
    )
    return listener, events


# ── Device bundle ───────────────────────────────────────────────────────────


@dataclass
class Device:
    """Per-frame XR -> JAKA glue. ``compute(obs)`` returns ``None`` while the clutch
    is disengaged so the loop can hold the measured pose."""

    compute: Callable[[dict | None], dict | None]
    startup: Callable[[], None]
    cleanup: Callable[[], None]
    telemetry: dict[str, Any] = field(default_factory=dict)


def build_device(cfg: "RecordConfig") -> tuple[JakaRobot, Device]:
    """Connect the JAKA arm and build the XR -> JAKA device bundle.

    Connects the follower FIRST so the clutch-home seed (in ``device.startup``) can
    read the live EE pose. On any failure after ``connect()`` the follower is
    disconnected so the connection never leaks.
    """
    if cfg.teleop.cloudxr_env_file is None:
        cfg.teleop.cloudxr_env_file = CLOUDXR_ENV_FILE

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    device: Device | None = None
    try:
        if not isinstance(robot, JakaRobot):
            raise ValueError(
                f"isaac_teleop_to_jaka.record requires --robot.type=jaka_robot, got {type(robot).__name__}"
            )
        if robot.config.user_frame_id != 0:
            raise ValueError(
                "isaac_teleop_to_jaka.record requires --robot.user_frame_id=0 because the "
                "default XR transform targets the robot base frame."
            )
        robot.config.control_mode = "ee_pose"
        # Invalidate cached feature dicts since control_mode changed.
        for cached in ("action_features", "observation_features"):
            robot.__dict__.pop(cached, None)

        bundle = make_xr_device(robot, cfg.teleop)
        device = Device(
            compute=bundle["compute"],
            startup=bundle["startup"],
            cleanup=bundle["cleanup"],
            telemetry=bundle["telemetry"],
        )
        device.startup()
    except BaseException:
        if device is not None:
            with suppress(Exception):
                device.cleanup()
        robot.disconnect()
        raise

    return robot, device


# ── Control panel ───────────────────────────────────────────────────────────


def _rounded(values: object | None) -> list[float] | None:
    if values is None:
        return None
    return [round(float(v), 4) for v in values]  # type: ignore[union-attr]


def _control_panel(telemetry: dict[str, Any], action: dict[str, float], obs: dict, frame_ms: float) -> Panel:
    start = time.perf_counter()
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column()

    table.add_row(Text("XR hand", style="bold bright_yellow"), "")
    table.add_row("grip_pos_m", str(_rounded(telemetry.get("grip_pos"))))
    table.add_row("grip_quat_xyzw", str(_rounded(telemetry.get("grip_quat"))))
    table.add_row("squeeze", str(round(float(telemetry.get("squeeze", 0.0)), 4)))
    table.add_row("trigger", str(round(float(telemetry.get("trigger", 0.0)), 4)))
    table.add_row("clutch_engaged", str(telemetry.get("clutch_engaged")))

    position = {axis: action[f"ee.{axis}"] for axis in ("x", "y", "z") if f"ee.{axis}" in action}
    orientation = {axis: action[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw") if f"ee.{axis}" in action}

    table.add_row("", "")
    table.add_row(Text("JAKA command", style="bold bright_green"), "")
    table.add_row("ee_target.position_m", str({k: round(v, 4) for k, v in position.items()}))
    table.add_row("ee_target.orientation_rad", str({k: round(v, 4) for k, v in orientation.items()}))
    if "gripper.pos" in action:
        table.add_row("gripper_pos", str(round(float(action["gripper.pos"]), 4)))

    # Actual measured EE pose, read from the robot's get_observation(). JAKA exposes
    # the same ee.* keys (m + rad) as the action, so we can diff against the command
    # directly. Missing keys (e.g. right after connect) degrade gracefully.
    actual_pos = {axis: obs[f"ee.{axis}"] for axis in ("x", "y", "z") if f"ee.{axis}" in obs}
    actual_ori = {axis: obs[f"ee.{axis}"] for axis in ("roll", "pitch", "yaw") if f"ee.{axis}" in obs}
    pos_delta = {k: round(actual_pos[k] - position[k], 4) for k in actual_pos if k in position}
    ori_delta = {
        k: round((actual_ori[k] - orientation[k] + math.pi) % (2 * math.pi) - math.pi, 4)
        for k in actual_ori
        if k in orientation
    }
    table.add_row("", "")
    table.add_row(Text("JAKA actual", style="bold bright_cyan"), "")
    table.add_row("ee_actual.position_m", str({k: round(v, 4) for k, v in actual_pos.items()}))
    table.add_row("ee_actual.orientation_rad", str({k: round(v, 4) for k, v in actual_ori.items()}))
    if pos_delta:
        table.add_row("delta.position_m", str(pos_delta))
    if ori_delta:
        table.add_row("delta.orientation_rad", str(ori_delta))

    panel_ms = (time.perf_counter() - start) * 1000
    panel = Panel(
        table,
        title="[bold cyan]Control frame[/bold cyan]",
        border_style="cyan",
        subtitle=f"[dim]panel: {panel_ms:.2f} ms | frame: {frame_ms:.2f} ms[/dim]",
    )
    return panel


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class RecordConfig:
    """CLI config for Isaac Teleop -> JAKA Cartesian dataset recording."""

    robot: JakaRobotConfig
    teleop: IsaacTeleopConfig
    dataset: DatasetRecordConfig

    # Resume recording on an existing (previously interrupted) dataset.
    resume: bool = False


# ── Record loop ─────────────────────────────────────────────────────────────


@safe_stop_image_writer
def _record_loop(
    robot: JakaRobot,
    device: Device,
    action_keys: list[str],
    events: dict,
    fps: int,
    live: Live,
    dataset: LeRobotDataset | None = None,
    control_time_s: float = 0.0,
    single_task: str | None = None,
) -> None:
    """Run one episode or reset phase of the Cartesian control loop.

    When ``dataset`` is None the loop still controls the robot so the operator can
    reposition during reset, but frames are not recorded.
    """
    control_interval = 1.0 / fps
    hold = HoldLatch(action_keys)
    start_t = time.perf_counter()
    timestamp = 0.0
    record_frames = dataset is not None

    while timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()

        if record_frames:
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

        # XR clutch disengaged: hold the TCP pose latched on the idle edge.
        raw = device.compute(obs)
        action = hold.resolve(raw, obs)

        live.update(_control_panel(device.telemetry, action, obs), refresh=True)

        sent_action = robot.send_action(action)

        if record_frames:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **action_frame, "task": single_task})

        # Work time of this iteration: obs read + compute + send + record. Excludes
        # the panel render (sub-ms) and the precise_sleep. If frame_ms ≫ 33 ms the
        # loop is missing servo ticks and the JAKA queue is growing.
        frame_ms = (time.perf_counter() - loop_start) * 1000
        live.update(_control_panel(device.telemetry, action, obs, frame_ms), refresh=True)

        precise_sleep(max(control_interval - (time.perf_counter() - loop_start), 0.0))
        timestamp = time.perf_counter() - start_t


# ── Entry point ─────────────────────────────────────────────────────────────


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    cfg.robot.control_mode = "ee_pose"

    nominal_fps = 1.0 / (cfg.robot.servo_step_num * 0.008)
    relative_fps_error = abs(float(cfg.dataset.fps) - nominal_fps) / nominal_fps
    if relative_fps_error > 0.1:
        raise ValueError(
            f"--dataset.fps={cfg.dataset.fps} does not match JAKA Servo Move "
            f"step_num={cfg.robot.servo_step_num} ({nominal_fps:.2f} Hz); "
            "keep the difference within 10% (use fps=30 with step_num=4)."
        )

    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot, device = build_device(cfg)

    # Dataset features: JAKA's fixed joint + gripper + EE action schema and
    # joint + TCP + EE observation schema. No host-side IK processor, so the
    # pipeline-aggregation step is a no-op and we can build the feature dict
    # directly from the robot's schemas.
    dataset_features = combine_feature_dicts(
        create_initial_features(action=robot.action_features),
        create_initial_features(observation=robot.observation_features),
    )

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    image_writer_threads = cfg.dataset.num_image_writer_threads_per_camera * num_cameras

    dataset: LeRobotDataset | None = None
    listener = None
    try:
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=image_writer_threads if num_cameras > 0 else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=image_writer_threads,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        listener, events = init_keyboard_listener()

        action_keys = sorted(robot.action_features.keys())

        loop_kwargs = {
            "robot": robot,
            "device": device,
            "action_keys": action_keys,
            "events": events,
            "fps": cfg.dataset.fps,
            "live": None,  # bound below
            "single_task": cfg.dataset.single_task,
        }

        initial_panel = Panel(
            "Waiting for the first control frame...",
            title="[bold cyan]Control frame[/bold cyan]",
            border_style="cyan",
        )
        with (
            Live(initial_panel, refresh_per_second=max(cfg.dataset.fps, 1), transient=False) as live,
            VideoEncodingManager(dataset),
        ):
            loop_kwargs["live"] = live
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                logging.info(f"Recording episode {dataset.num_episodes}")
                _record_loop(
                    **loop_kwargs,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                )

                # Reset window: give the operator time to reposition the scene.
                # Skipped for the last episode (or if stop_recording was set).
                if not events["stop_recording"] and (
                    recorded_episodes < cfg.dataset.num_episodes - 1 or events["rerecord_episode"]
                ):
                    logging.info("Reset the environment")
                    _record_loop(
                        **loop_kwargs,
                        dataset=None,
                        control_time_s=cfg.dataset.reset_time_s,
                    )

                if events["rerecord_episode"]:
                    logging.info("Re-record episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        logging.info("Stop recording")

        # Hardware teardown FIRST, each step guarded: the arm must be freed
        # promptly (not after a potentially long finalize/encode), a cleanup
        # failure must not skip the follower disconnect, and neither must
        # prevent the dataset from being finalized below.
        with suppress(Exception):
            device.cleanup()
        with suppress(Exception):
            if robot.is_connected:
                robot.disconnect()

        # Restore the terminal before the (potentially long) finalize/encode.
        if listener is not None:
            with suppress(Exception):
                listener.stop()

        if dataset is not None:
            dataset.finalize()

        if cfg.dataset.push_to_hub:
            if dataset is not None and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        logging.info("Exiting")

    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
