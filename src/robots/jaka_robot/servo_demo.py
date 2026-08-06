"""Standalone JAKA ``servo_p`` Cartesian motion demo.

This file intentionally does not import the LeRobot adapter or any other
repository module.  It talks to the JAKA Python SDK directly and is intended
to be run manually with a clear workspace and an accessible emergency stop::

    python src/robots/jaka_robot/servo_demo.py --ip 192.168.1.31

The SDK uses millimetres for Cartesian positions.  By default the TCP moves
smoothly along base-frame Z by +/- 50 mm (about 5 cm from its measured start
position) at the documented 8 ms ``servo_p`` period.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import math
import sys
import time
from pathlib import Path
from typing import Any

ABS = 0
SERVO_PERIOD_S = 0.008


def _load_jkrc() -> Any:
    """Load the SDK bundled next to this script, without repository imports."""

    sdk_dir = Path(__file__).resolve().parent
    api_library = sdk_dir / "libjakaAPI.so"
    if api_library.is_file():
        ctypes.CDLL(str(api_library), mode=ctypes.RTLD_GLOBAL)
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    try:
        return importlib.import_module("jkrc")
    except ImportError as exc:
        raise RuntimeError(f"Unable to load JAKA Python SDK from {sdk_dir}: {exc}") from exc


def _payload(operation: str, result: Any) -> tuple[Any, ...]:
    """Validate an SDK return value and return its payload."""

    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"{operation} returned an invalid value: {result!r}")
    try:
        code = int(result[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{operation} returned an invalid status: {result!r}") from exc
    if code != 0:
        raise RuntimeError(f"{operation} failed with SDK error {code}: {result!r}")
    values = tuple(result[1:])
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = tuple(values[0])
    return values


def _servo_p(robot: Any, pose: list[float]) -> int | None:
    """Send one absolute Cartesian frame, accepting SDK 1.7.2 signatures."""

    pose_tuple = tuple(float(value) for value in pose)
    # SDK 1.7.2 documents do_info as an optional fourth argument.  Some
    # installed Python bindings expose only the original three-argument form.
    try:
        result = robot.servo_p(pose_tuple, ABS, 1, None)
    except TypeError:
        result = robot.servo_p(pose_tuple, ABS, 1)
    values = _payload("servo_p", result)
    if not values:
        return None
    try:
        queue_depth = int(values[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"servo_p returned an invalid queue depth: {result!r}") from exc
    if not 0 <= queue_depth <= 100:
        raise RuntimeError(f"servo_p returned an invalid queue depth: {result!r}")
    return queue_depth


def _controller_state(robot: Any) -> tuple[bool, bool]:
    getter = getattr(robot, "get_robot_status_simple", None)
    if getter is None:
        # The demo can still run with older bindings; in that case the calls
        # below are harmless and cleanup treats both states as script-owned.
        return False, False
    values = _payload("get_robot_status_simple", getter())
    if len(values) < 2:
        raise RuntimeError(f"get_robot_status_simple returned too few values: {values!r}")
    return bool(values[-2]), bool(values[-1])


def _get_tcp_pose(robot: Any) -> list[float]:
    getter = getattr(robot, "get_tcp_position", None) or getattr(robot, "get_actual_tcp_position", None)
    if getter is None:
        raise RuntimeError("JAKA SDK does not expose get_tcp_position()")
    pose = list(_payload("get_tcp_position", getter()))
    if len(pose) != 6 or not all(math.isfinite(float(value)) for value in pose):
        raise RuntimeError(f"get_tcp_position returned an invalid pose: {pose!r}")
    return [float(value) for value in pose]


def _wait_until(deadline: float) -> None:
    """Sleep until a cumulative deadline, avoiding drift between frames."""

    remaining = deadline - time.perf_counter()
    if remaining > 0.0005:
        time.sleep(remaining - 0.0005)
    while time.perf_counter() < deadline:
        pass


def run_demo(robot: Any, *, duration_s: float, amplitude_mm: float, frequency_hz: float) -> int:
    """Run the finite up/down trajectory and return the number of frames sent."""

    if not all(
        math.isfinite(float(value)) and float(value) > 0 for value in (duration_s, amplitude_mm, frequency_hz)
    ):
        raise ValueError("duration_s, amplitude_mm, and frequency_hz must be positive")

    base_pose = _get_tcp_pose(robot)
    frame_count = max(1, math.ceil(duration_s / SERVO_PERIOD_S))
    print(f"base TCP pose [mm/rad]: {base_pose}")
    print(
        f"servo_p: {frame_count} frames at {SERVO_PERIOD_S * 1000:.1f} ms, "
        f"Z amplitude +/-{amplitude_mm:.1f} mm, frequency {frequency_hz:.3f} Hz"
    )

    start = time.perf_counter()
    for index in range(frame_count):
        elapsed = index * SERVO_PERIOD_S
        pose = base_pose.copy()
        pose[2] += amplitude_mm * math.sin(2.0 * math.pi * frequency_hz * elapsed)
        queue_depth = _servo_p(robot, pose)
        if queue_depth is not None and queue_depth >= 80:
            raise RuntimeError(f"servo_p queue is nearly full ({queue_depth}/100)")
        _wait_until(start + (index + 1) * SERVO_PERIOD_S)
    return frame_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone JAKA servo_p TCP up/down test")
    parser.add_argument("--ip", default="192.168.1.31", help="JAKA controller IP address")
    parser.add_argument("--duration-s", type=float, default=20.0, help="test duration (default: 20 s)")
    parser.add_argument(
        "--amplitude-mm", type=float, default=50.0, help="peak Z displacement in mm (default: 50 mm)"
    )
    parser.add_argument("--frequency-hz", type=float, default=0.1, help="up/down frequency (default: 0.1 Hz)")
    parser.add_argument("--dry-run", action="store_true", help="validate arguments without connecting")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not all(
        math.isfinite(float(value)) and float(value) > 0
        for value in (args.duration_s, args.amplitude_mm, args.frequency_hz)
    ):
        raise SystemExit("duration, amplitude, and frequency must be positive")
    if args.dry_run:
        print(
            f"dry run: duration={args.duration_s:.1f}s, amplitude=+/-{args.amplitude_mm:.1f}mm, "
            f"frequency={args.frequency_hz:.3f}Hz"
        )
        return

    print("WARNING: this script controls a physical JAKA robot through servo_p.")
    print("Check the workspace, tool/user frames, payload, limits, and emergency stop.")
    if input("Type SERVO_P to connect and continue: ").strip() != "SERVO_P":
        print("Cancelled before connecting.")
        return

    jkrc = _load_jkrc()
    robot = jkrc.RC(args.ip)
    powered_by_script = enabled_by_script = servo_enabled = False
    try:
        _payload("login", robot.login())
        initially_powered, initially_enabled = _controller_state(robot)
        if not initially_powered:
            _payload("power_on", robot.power_on())
            powered_by_script = True
        if not initially_enabled:
            _payload("enable_robot", robot.enable_robot())
            enabled_by_script = True

        _payload("servo_move_enable", robot.servo_move_enable(True, True))
        servo_enabled = True
        frames = run_demo(
            robot,
            duration_s=args.duration_s,
            amplitude_mm=args.amplitude_mm,
            frequency_hz=args.frequency_hz,
        )
        print(f"Completed servo_p test ({frames} frames).")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if servo_enabled:
            try:
                _payload("servo_move_enable", robot.servo_move_enable(False, True))
            except Exception as exc:
                print(f"Warning: could not disable servo mode: {exc}")
        if enabled_by_script:
            try:
                _payload("disable_robot", robot.disable_robot())
            except Exception as exc:
                print(f"Warning: could not disable robot: {exc}")
        if powered_by_script:
            try:
                _payload("power_off", robot.power_off())
            except Exception as exc:
                print(f"Warning: could not power off robot: {exc}")
        logout = getattr(robot, "logout", None) or getattr(robot, "log_out", None)
        if logout is not None:
            try:
                _payload("logout", logout())
            except Exception as exc:
                print(f"Warning: could not logout: {exc}")


if __name__ == "__main__":
    main()
