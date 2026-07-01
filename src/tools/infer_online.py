import time
from dataclasses import asdict, dataclass

import draccus
import numpy as np
from rich import print  # pip install rich
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from lerobot.async_inference.client import PolicyClient, PolicyClientConfig

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.ros_robot import RosRobot, RosRobotConfig
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.rerun_utils import RerunLogger


def _build_timing_table(
    latest_ms: dict[str, float], avg_ms: dict[str, float], loops: int, target_fps: int, actions_str: str = ""
) -> Group:
    table = Table(title=f"Control Loop Timing (iterations={loops}, target_fps={target_fps})")
    table.add_column("Stage")
    table.add_column("Latest (ms)", justify="right")
    table.add_column("Average (ms)", justify="right")

    rows = [
        ("send_action", "send_action_ms"),
        ("dataset_add", "dataset_add_ms"),
        ("rerun_log", "rerun_log_ms"),
        ("pause_wait", "pause_wait_ms"),
        ("busy_wait", "busy_wait_ms"),
        ("total", "total_ms"),
    ]
    for label, key in rows:
        table.add_row(label, f"{latest_ms[key]:.3f}", f"{avg_ms[key]:.3f}")

    latest_fps = 1000.0 / latest_ms["total_ms"] if latest_ms["total_ms"] > 0 else float("inf")
    avg_fps = 1000.0 / avg_ms["total_ms"] if avg_ms["total_ms"] > 0 else float("inf")
    table.add_row("fps", f"{latest_fps:.2f}", f"{avg_fps:.2f}")
    return Group(Text(f"actions: {actions_str}"), table)


@dataclass
class PlayConfig:
    # Task string sent to the policy (e.g. "pick up the bin")
    task: str
    # Limit the frames per second. By default, uses the policy fps.
    fps: int
    # Robot configuration, support
    robot: RosRobotConfig
    # Policy configuration
    policy: PolicyClientConfig
    # Optional: Remote Rerun viewer URL for visualization. If None, no visualization.
    rerun_url: str | None = None
    # Optional: Record the inference process.
    record: bool = False


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {
        "play": False,
        "stop": False,
        "forward": False,
        "policy_control": True,
    }

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                events["play"] = not events["play"]
                print(f"[yellow]Toggled play: {'▶️' if events['play'] else '⏸️'}[/yellow]")
            elif key == keyboard.Key.esc:
                events["stop"] = True
                print("[red]Pressed [ESC]. Bye![/red]")
            elif key == keyboard.Key.right:
                events["forward"] = True
                events["play"] = False
                print("[yellow]Pressed ➡️. Forward to next episode[/yellow]")

        except Exception as e:
            print(f"[red]Error handling key press: {e}[/red]")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def play_episode(
    robot: RosRobot,
    policy: PolicyClient,
    fps: int,
    events: dict,
    logger: RerunLogger = None,
    dataset: LeRobotDataset | None = None,
):
    policy.reset()
    start_episode_t = time.perf_counter()
    dt = 1.0 / fps
    latest_ms = {
        "send_action_ms": 0.0,
        "dataset_add_ms": 0.0,
        "rerun_log_ms": 0.0,
        "pause_wait_ms": 0.0,
        "busy_wait_ms": 0.0,
        "total_ms": 0.0,
    }
    avg_ms = latest_ms.copy()
    loop_count = 0
    actions_str = ""

    # Start control loop
    with Live(
        _build_timing_table(latest_ms, avg_ms, loop_count, fps, actions_str),
        refresh_per_second=8,
        transient=True,
    ) as live:
        while True:
            # Start control loop
            start = time.perf_counter()

            # Request action from policy
            actions = policy.require_action().numpy()
            with np.printoptions(precision=2, suppress=True, linewidth=999):
                actions_str = f"{actions[0:8]} | {actions[8:16]}"
                # print(actions_str)

            # Send action to the robot
            t0 = time.perf_counter()
            policy_action = robot.send_action(actions)
            latest_ms["send_action_ms"] = (time.perf_counter() - t0) * 1000.0

            observation = policy.last_observation

            t0 = time.perf_counter()
            if dataset is not None and observation is not None:
                dataset.add_frame(
                    {**observation, **policy_action, **robot.get_control_mode(), "task": robot.task}
                )
            latest_ms["dataset_add_ms"] = (time.perf_counter() - t0) * 1000.0

            # Log observation and action to Rerun
            t0 = time.perf_counter()
            if logger and observation is not None:
                logger.log({**observation, "policy": actions})
            latest_ms["rerun_log_ms"] = (time.perf_counter() - t0) * 1000.0

            # Check if user requested to stop or forward
            if events.get("forward", False) or events.get("stop", False):
                latest_ms["pause_wait_ms"] = 0.0
                latest_ms["busy_wait_ms"] = 0.0
                latest_ms["total_ms"] = (time.perf_counter() - start) * 1000.0
                loop_count += 1
                for key, value in latest_ms.items():
                    avg_ms[key] = (avg_ms[key] * (loop_count - 1) + value) / loop_count
                live.update(_build_timing_table(latest_ms, avg_ms, loop_count, fps, actions_str))
                break

            # Check if user requested to pause
            t0 = time.perf_counter()
            while not events["play"] and not events["stop"] and not events["forward"]:
                time.sleep(0.01)
            latest_ms["pause_wait_ms"] = (time.perf_counter() - t0) * 1000.0

            # Control loop timing to maintain the desired fps
            t0 = time.perf_counter()
            busy_wait(max(0.0, dt - (time.perf_counter() - start)))
            latest_ms["busy_wait_ms"] = (time.perf_counter() - t0) * 1000.0
            latest_ms["total_ms"] = (time.perf_counter() - start) * 1000.0

            loop_count += 1
            for key, value in latest_ms.items():
                avg_ms[key] = (avg_ms[key] * (loop_count - 1) + value) / loop_count

            live.update(_build_timing_table(latest_ms, avg_ms, loop_count, fps, actions_str))

    # Return the total time taken for the episode
    return time.perf_counter() - start_episode_t


@safe_disconnect
def play(robot: RosRobot, policy: PolicyClient, config: PlayConfig) -> LeRobotDataset:

    assert robot.is_connected, "Robot should be connected before recording."

    # Create dataset to store the episode data
    if config.record:
        repo_id = f"{policy.policy_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=f"data/{repo_id}",
            fps=config.fps,
            robot=robot,
            batch_encoding_size=10,
        )
        print(f"[green]Dataset created with repo_id: {repo_id}[/green]")
        print(f"Dataset Features: {dataset.features.keys()}")
    else:
        dataset = None

    _, events = init_keyboard_listener()
    print("[yellow]Please use keyboard to recording...[/yellow]")

    logger = RerunLogger(url=config.rerun_url) if config.rerun_url else None

    while True:
        print("[yellow]Please wait for the robot to be ready[/yellow]")
        robot.reset()
        # Wait for the user to press the start button
        print(
            "[bright_yellow]Keyboard usage: [SPACE] to play/pause, [ESC] to stop, [→] to forward[/bright_yellow]"
        )
        # Reset events
        events["forward"] = False
        # Wait for user to start or stop, you should wait for robot to be ready
        while events and not events["play"] and not events["stop"]:
            time.sleep(0.1)

        # Check if stop requested
        if events["stop"]:
            print("Bye!")
            break

        if logger:
            logger.switch_record()

        # Start play an episode
        elapsed = play_episode(
            robot=robot,
            policy=policy,
            events=events,
            fps=config.fps,
            logger=logger,
            dataset=dataset,
        )

        # Print episode summary
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"[bright_yellow]Episode finished in {formatted_time}[/bright_yellow]")

        # Pause after each episode
        events["play"] = False

        # Finished recording the episode
        if dataset is not None and dataset.episode_buffer["size"] > 0:
            dataset.save_episode()

    if logger:
        logger.stop()

    if dataset is not None:
        dataset._batch_encode_episode_video(dataset.num_episodes - 1)

    print("[yellow]The show is over.[/yellow]")


@draccus.wrap()
def main(config: PlayConfig):
    print(asdict(config))

    robot = RosRobot(config.robot)
    robot.set_task(config.task)
    robot.connect()
    # robot.run_test(num_iterations=100)

    policy = PolicyClient(config.policy, obs_fn=robot.capture_observation)

    play(robot, policy, config)

    robot.disconnect()


if __name__ == "__main__":
    main()
