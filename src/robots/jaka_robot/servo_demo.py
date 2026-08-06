"""Manual JAKA Servo Move demonstration.

WARNING: This script powers on, enables, and moves a real robot. Automation
must not run it. Run it manually only after checking the workspace, tool and user
frames, payload, collision settings, and emergency stop.
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np

from robots.jaka_robot.jaka_robot import JakaError, JakaRobot, JakaRobotConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually test managed JAKA Servo target tracking")
    parser.add_argument("--ip", default="192.168.1.31", help="JAKA controller IP")
    parser.add_argument("--mode", choices=("eef", "joint"), default="eef")
    parser.add_argument("--duration", type=float, default=8.0, help="trajectory duration in seconds")
    parser.add_argument("--target-hz", type=float, default=30.0, help="desired-target update rate")
    return parser.parse_args()


def _run_stream(robot: JakaRobot, mode: str, duration_s: float, target_hz: float) -> None:
    if duration_s <= 0 or target_hz <= 0:
        raise ValueError("duration and target_hz must be positive")

    if mode == "eef":
        initial = robot.get_eef_pose()
        amplitude = 0.005  # 5 mm along base Z.
        frequency_hz = 0.25

        def send_target(elapsed_s: float) -> None:
            target = initial.copy()
            target[2] += amplitude * math.sin(2.0 * math.pi * frequency_hz * elapsed_s)
            robot.send_action(
                dict(zip(("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw"), target, strict=True)),
                use_servo=True,
            )

    else:
        initial = robot.get_joint_positions()
        amplitude = 0.01  # radians on joint 1.
        frequency_hz = 0.25

        def send_target(elapsed_s: float) -> None:
            target = initial["joint_1.pos"] + amplitude * math.sin(2.0 * math.pi * frequency_hz * elapsed_s)
            robot.send_action({"joint_1.pos": target}, use_servo=True)

    latencies: list[float] = []
    periods: list[float] = []
    overruns = 0
    period_s = 1.0 / target_hz
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
        send_target(elapsed)
        latencies.append(time.perf_counter() - call_start)

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
    print(f"targets updated: {len(latencies)}")
    print(f"target update frequency: {achieved_hz:.2f} Hz")
    print(f"target update overruns: {overruns}")
    if latency_ms.size:
        print(
            f"SDK latency mean/p95/max: {latency_ms.mean():.2f}/{np.percentile(latency_ms, 95):.2f}/{latency_ms.max():.2f} ms"
        )
    if period_ms.size:
        print(f"frame period mean/p95: {period_ms.mean():.2f}/{np.percentile(period_ms, 95):.2f} ms")
    status = robot.get_servo_status()
    print(
        "managed sender frequency/p95/max: "
        f"{status['send_rate_hz']:.2f} Hz/{status['period_p95_ms']:.2f} ms/"
        f"{status['period_max_ms']:.2f} ms"
    )
    print(f"managed sender overruns: {status['overruns']}; queue depth: {status['queue_depth']}")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    config = JakaRobotConfig(
        ip=args.ip,
        auto_power_on=True,
        auto_enable=True,
    )

    print("WARNING: this program will power on, enable, and move a real JAKA robot.")
    print(f"mode={args.mode}, duration={args.duration:.1f}s, target_hz={args.target_hz:.1f}")
    if input("After checking the workspace and emergency stop, type SERVO: ").strip() != "SERVO":
        print("Servo test cancelled before connecting to the robot.")
        return

    robot = JakaRobot(config)
    try:
        with robot:
            _run_stream(robot, args.mode, args.duration, args.target_hz)
            robot.servo_enable(False)
            print("Servo Move test complete; servo mode has been disabled.")
    except (JakaError, KeyboardInterrupt) as e:
        print(f"Servo Move test interrupted: {e}")
        raise


if __name__ == "__main__":
    main()
