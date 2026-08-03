"""Manual JAKA Servo Move demonstration.

WARNING: This script powers on, enables, and moves a real robot. Claude must
not run it. Run it manually only after checking the workspace, tool and user
frames, payload, collision settings, and emergency stop.
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np

from hardwares.jaka_robot.jaka_robot import ABS, JakaError, JakaRobot, JakaRobotConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually test JAKA Servo Move at a nominal >=30 Hz")
    parser.add_argument("--ip", default="192.168.1.31", help="JAKA controller IP")
    parser.add_argument("--mode", choices=("eef", "joint"), default="eef")
    parser.add_argument("--duration", type=float, default=8.0, help="trajectory duration in seconds")
    parser.add_argument("--step-num", type=int, default=4, help="controller cycles per frame (1..4)")
    return parser.parse_args()


def _run_stream(robot: JakaRobot, mode: str, duration_s: float, step_num: int) -> None:
    if duration_s <= 0:
        raise ValueError("duration must be positive")

    if mode == "eef":
        initial = robot.get_eef_pose()
        amplitude = 0.005  # 5 mm along base Z.
        frequency_hz = 0.25

        def send_target(elapsed_s: float) -> int | None:
            target = initial.copy()
            target[2] += amplitude * math.sin(2.0 * math.pi * frequency_hz * elapsed_s)
            return robot.servo_eef_frame(target, move_mode=ABS, step_num=step_num)

    else:
        initial = np.asarray(
            robot._check(robot.rc.get_actual_joint_position(), return_payload=True),
            dtype=float,
        )
        amplitude = 0.01  # radians on joint 1.
        frequency_hz = 0.25

        def send_target(elapsed_s: float) -> int | None:
            target = initial.copy()
            target[0] += amplitude * math.sin(2.0 * math.pi * frequency_hz * elapsed_s)
            return robot.servo_joint_frame(target, move_mode=ABS, step_num=step_num)

    latencies: list[float] = []
    periods: list[float] = []
    queue_depths: list[int] = []
    overruns = 0

    with robot.servo_stream(step_num=step_num) as period_s:
        start = previous = time.perf_counter()
        deadline = start
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= duration_s:
                break
            periods.append(now - previous)
            previous = now

            call_start = time.perf_counter()
            queue_depth = send_target(elapsed)
            latencies.append(time.perf_counter() - call_start)
            if queue_depth is not None:
                queue_depths.append(queue_depth)

            deadline += period_s
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                overruns += 1
                deadline = time.perf_counter()

    elapsed_total = time.perf_counter() - start
    achieved_hz = len(latencies) / elapsed_total if elapsed_total else 0.0
    latency_ms = np.asarray(latencies) * 1000.0
    period_ms = np.asarray(periods[1:]) * 1000.0
    print(f"frames sent: {len(latencies)}")
    print(f"nominal frequency: {1.0 / period_s:.2f} Hz")
    print(f"measured frequency: {achieved_hz:.2f} Hz")
    print(f"deadline overruns: {overruns}")
    if latency_ms.size:
        print(f"SDK latency mean/p95/max: {latency_ms.mean():.2f}/{np.percentile(latency_ms, 95):.2f}/{latency_ms.max():.2f} ms")
    if period_ms.size:
        print(f"frame period mean/p95: {period_ms.mean():.2f}/{np.percentile(period_ms, 95):.2f} ms")
    if queue_depths:
        print(f"maximum reported queue depth: {max(queue_depths)}")
    if achieved_hz < 30.0:
        print("WARNING: measured control frequency is below the required 30 Hz.")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    config = JakaRobotConfig(
        ip=args.ip,
        auto_power_on=True,
        auto_enable=True,
        servo_step_num=args.step_num,
    )

    print("WARNING: this program will power on, enable, and move a real JAKA robot.")
    print(f"mode={args.mode}, duration={args.duration:.1f}s, step_num={args.step_num}")
    if input("After checking the workspace and emergency stop, type SERVO: ").strip() != "SERVO":
        print("Servo test cancelled before connecting to the robot.")
        return

    robot = JakaRobot(config)
    try:
        with robot:
            _run_stream(robot, args.mode, args.duration, args.step_num)
            print("Servo Move test complete; servo mode has been disabled.")
    except (JakaError, KeyboardInterrupt) as e:
        print(f"Servo Move test interrupted: {e}")
        raise


if __name__ == "__main__":
    main()
