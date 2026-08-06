"""Isolated raw ``servo_p`` diagnostic for JAKA SDK 1.7.2.

This script intentionally bypasses :class:`JakaRobot`'s managed Servo sender.
It holds the measured TCP pose or sends one tiny minimum-jerk Cartesian bump
while recording the timing data needed to distinguish trajectory, scheduler,
network, and controller-queue problems.

It controls physical hardware. Run it only with a clear workspace and an
accessible emergency stop.

# Run hold mode for 5 seconds and write CSV:

uv run python -m robots.jaka_robot.servo_p_debug \
  --ip 192.168.1.31 \
  --mode hold \
  --duration-s 5 \
  --csv artifacts/servo_p_hold.csv


# Move the TCP in a 1 mm Z bump over 4 seconds, with 1 second of settling before and after, and write CSV:

uv run python -m robots.jaka_robot.servo_p_debug \
  --ip 192.168.1.31 \
  --mode z-bump \
  --duration-s 4 \
  --settle-s 1 \
  --amplitude-mm 1 \
  --csv artifacts/servo_p_z_bump.csv

"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .jaka_robot import ABS, SERVO_CYCLE_S, JakaError, _payload, _vector, create_rc

logger = logging.getLogger(__name__)

_Mode = Literal["hold", "z-bump"]


@dataclass(frozen=True)
class DiagnosticSettings:
    mode: _Mode = "hold"
    duration_s: float = 5.0
    settle_s: float = 1.0
    amplitude_mm: float = 1.0
    period_s: float = SERVO_CYCLE_S
    queue_abort_depth: int = 80
    max_consecutive_overruns: int = 5
    overrun_factor: float = 1.5
    spin_threshold_s: float = 0.0005

    def validate(self) -> None:
        if self.duration_s <= 0 or self.settle_s < 0 or self.amplitude_mm <= 0:
            raise ValueError("duration and amplitude must be positive; settle must not be negative")
        if self.period_s <= 0:
            raise ValueError("period must be positive")
        if not 1 <= self.queue_abort_depth <= 100:
            raise ValueError("queue_abort_depth must be in [1, 100]")
        if self.max_consecutive_overruns < 1 or self.overrun_factor <= 1.0:
            raise ValueError("overrun watchdog settings are invalid")
        if not 0 <= self.spin_threshold_s < self.period_s:
            raise ValueError("spin_threshold_s must be in [0, period)")


@dataclass(frozen=True)
class ServoSample:
    index: int
    scheduled_s: float
    started_s: float
    period_s: float
    lateness_s: float
    sdk_latency_s: float
    queue_depth: int | None
    target: tuple[float, float, float, float, float, float]


class ServoPDiagnosticError(RuntimeError):
    """A watchdog or SDK failure with all samples captured before it."""

    def __init__(self, message: str, samples: Sequence[ServoSample]):
        super().__init__(message)
        self.samples = tuple(samples)


def _minimum_jerk(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def build_targets(initial_pose_mm: Sequence[float], settings: DiagnosticSettings) -> list[np.ndarray]:
    """Build all targets before entering Servo Move so the loop does no planning."""

    settings.validate()
    initial = _vector(initial_pose_mm, name="initial TCP pose")
    if settings.mode == "hold":
        frame_count = max(2, math.ceil(settings.duration_s / settings.period_s) + 1)
        return [initial.copy() for _ in range(frame_count)]

    settle_frames = math.ceil(settings.settle_s / settings.period_s)
    motion_frames = max(3, math.ceil(settings.duration_s / settings.period_s) + 1)
    targets = [initial.copy() for _ in range(settle_frames)]
    for index in range(motion_frames):
        progress = index / (motion_frames - 1)
        half_progress = 2.0 * progress if progress <= 0.5 else 2.0 * (1.0 - progress)
        target = initial.copy()
        target[2] += settings.amplitude_mm * _minimum_jerk(half_progress)
        targets.append(target)
    targets.extend(initial.copy() for _ in range(settle_frames))
    return targets


def _wait_until(
    deadline: float,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    spin_threshold_s: float,
) -> None:
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return
        if remaining > spin_threshold_s:
            sleep(remaining - spin_threshold_s)
            continue
        # The final sub-millisecond spin avoids relying on scheduler wake-up
        # precision. No SDK call, logging, allocation, or file IO occurs here.
        while clock() < deadline:
            pass
        return


def _servo_p_call(rc: Any, target: np.ndarray, *, robot_id: int) -> int | None:
    pose = tuple(map(float, target))
    try:
        result = rc.servo_p(pose, ABS, 1) if robot_id == 0 else rc.servo_p(pose, ABS, 1, None, robot_id)
    except TypeError as exc:
        if robot_id != 0:
            raise RuntimeError(
                "Installed JAKA Python binding does not accept servo_p(..., do_info, robot_id); use --robot-id 0"
            ) from exc
        raise
    payload = _payload("servo_p", result)
    if not payload:
        return None
    try:
        depth = int(payload[0])
    except (TypeError, ValueError) as exc:
        raise JakaError("servo_p", -1, payload) from exc
    if not 0 <= depth <= 100:
        raise JakaError("servo_p", -1, payload)
    return depth


def run_stream(
    rc: Any,
    targets: Sequence[np.ndarray],
    settings: DiagnosticSettings,
    *,
    robot_id: int = 0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ServoSample]:
    """Send precomputed targets at cumulative 8 ms deadlines."""

    settings.validate()
    if not targets:
        raise ValueError("targets must not be empty")

    samples: list[ServoSample] = []
    start = clock()
    previous_start = start
    consecutive_overruns = 0
    for index, target in enumerate(targets):
        deadline = start + index * settings.period_s
        _wait_until(
            deadline,
            clock=clock,
            sleep=sleep,
            spin_threshold_s=settings.spin_threshold_s,
        )
        call_started = clock()
        period = 0.0 if index == 0 else call_started - previous_start
        previous_start = call_started
        lateness = max(0.0, call_started - deadline)

        sdk_started = clock()
        try:
            queue_depth = _servo_p_call(rc, _vector(target, name="Servo target"), robot_id=robot_id)
        except Exception as exc:
            raise ServoPDiagnosticError(f"frame {index} servo_p failed: {exc}", samples) from exc
        sdk_latency = clock() - sdk_started

        overrun_threshold = settings.period_s * settings.overrun_factor
        overrun = index > 0 and (period > overrun_threshold or sdk_latency > overrun_threshold)
        consecutive_overruns = consecutive_overruns + 1 if overrun else 0
        sample = ServoSample(
            index=index,
            scheduled_s=deadline - start,
            started_s=call_started - start,
            period_s=period,
            lateness_s=lateness,
            sdk_latency_s=sdk_latency,
            queue_depth=queue_depth,
            target=tuple(map(float, target)),
        )
        samples.append(sample)

        if queue_depth is not None and queue_depth >= settings.queue_abort_depth:
            raise ServoPDiagnosticError(
                f"servo_p queue depth reached {queue_depth}; abort threshold is {settings.queue_abort_depth}",
                samples,
            )
        if consecutive_overruns >= settings.max_consecutive_overruns:
            raise ServoPDiagnosticError(
                "servo_p timing watchdog saw "
                f"{consecutive_overruns} consecutive overruns at frame {index} "
                f"(period={period * 1000.0:.3f}ms, SDK latency={sdk_latency * 1000.0:.3f}ms, "
                f"threshold={overrun_threshold * 1000.0:.3f}ms)",
                samples,
            )
    return samples


def _set_servo_move(rc: Any, enabled: bool, *, robot_id: int) -> None:
    try:
        result = rc.servo_move_enable(enabled, True, robot_id)
    except TypeError:
        if robot_id != 0:
            raise RuntimeError(
                "Installed JAKA Python binding does not accept servo_move_enable(..., robot_id); use --robot-id 0"
            ) from None
        try:
            result = rc.servo_move_enable(enabled, True)
        except TypeError:
            result = rc.servo_move_enable(enabled)
    _payload("servo_move_enable", result)


def _login(rc: Any) -> None:
    login = getattr(rc, "login", None) or getattr(rc, "log_in", None)
    if login is None:
        raise RuntimeError("Installed JAKA binding does not expose login() or log_in()")
    try:
        result = login(use_grpc=0)
    except TypeError:
        result = login()
    _payload("login", result)


def _controller_state(rc: Any) -> tuple[bool, bool]:
    getter = getattr(rc, "get_robot_status_simple", None)
    if getter is None:
        return False, False
    payload = _payload("get_robot_status_simple", getter())
    if len(payload) < 2:
        raise JakaError("get_robot_status_simple", -1, payload)
    return bool(payload[-2]), bool(payload[-1])


def _write_csv(path: Path, samples: Sequence[ServoSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "index",
                "scheduled_s",
                "started_s",
                "period_ms",
                "lateness_ms",
                "sdk_latency_ms",
                "queue_depth",
                "x_mm",
                "y_mm",
                "z_mm",
                "roll_rad",
                "pitch_rad",
                "yaw_rad",
            )
        )
        for sample in samples:
            writer.writerow(
                (
                    sample.index,
                    sample.scheduled_s,
                    sample.started_s,
                    sample.period_s * 1000.0,
                    sample.lateness_s * 1000.0,
                    sample.sdk_latency_s * 1000.0,
                    sample.queue_depth,
                    *sample.target,
                )
            )


def _print_report(samples: Sequence[ServoSample], settings: DiagnosticSettings) -> None:
    if not samples:
        print("frames: 0; no successful servo_p sample was captured")
        return
    periods_ms = np.asarray([sample.period_s for sample in samples[1:]]) * 1000.0
    latencies_ms = np.asarray([sample.sdk_latency_s for sample in samples]) * 1000.0
    lateness_ms = np.asarray([sample.lateness_s for sample in samples]) * 1000.0
    queue_depths = [sample.queue_depth for sample in samples if sample.queue_depth is not None]
    overruns = int(np.count_nonzero(periods_ms > settings.period_s * settings.overrun_factor * 1000.0))
    elapsed = samples[-1].started_s + samples[-1].sdk_latency_s if samples else 0.0
    achieved_hz = (len(samples) - 1) / elapsed if elapsed > 0 and len(samples) > 1 else 0.0

    print(f"frames: {len(samples)}; elapsed: {elapsed:.3f}s; achieved: {achieved_hz:.2f} Hz")
    if periods_ms.size:
        print(
            "period ms mean/p95/max: "
            f"{periods_ms.mean():.3f}/{np.percentile(periods_ms, 95):.3f}/{periods_ms.max():.3f}"
        )
    print(
        "SDK latency ms mean/p95/max: "
        f"{latencies_ms.mean():.3f}/{np.percentile(latencies_ms, 95):.3f}/{latencies_ms.max():.3f}"
    )
    print(f"max deadline lateness: {lateness_ms.max():.3f} ms; period overruns: {overruns}")
    print(f"maximum queue depth: {max(queue_depths) if queue_depths else 'not reported'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated raw JAKA servo_p timing diagnostic")
    parser.add_argument("--ip", required=True, help="JAKA controller IP address")
    parser.add_argument("--mode", choices=("hold", "z-bump"), default="hold")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--settle-s", type=float, default=1.0, help="hold time before/after z-bump")
    parser.add_argument("--amplitude-mm", type=float, default=1.0, help="positive Z bump amplitude")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--queue-abort-depth", type=int, default=80)
    parser.add_argument("--max-consecutive-overruns", type=int, default=5)
    parser.add_argument("--power-on", action="store_true", help="power on when currently off")
    parser.add_argument("--enable", action="store_true", help="enable when currently disabled")
    parser.add_argument(
        "--leave-powered-enabled",
        action="store_true",
        help="do not restore power/enable state changed by this script",
    )
    parser.add_argument("--csv", type=Path, help="write samples after Servo Move exits")
    parser.add_argument("--dry-run", action="store_true", help="validate arguments without connecting")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = DiagnosticSettings(
        mode=args.mode,
        duration_s=args.duration_s,
        settle_s=args.settle_s,
        amplitude_mm=args.amplitude_mm,
        queue_abort_depth=args.queue_abort_depth,
        max_consecutive_overruns=args.max_consecutive_overruns,
    )
    settings.validate()
    if args.robot_id < 0:
        raise SystemExit("--robot-id must not be negative")
    if args.dry_run:
        print(f"dry run: {settings}")
        return

    print("WARNING: this diagnostic takes direct control of a physical JAKA robot through servo_p.")
    print("Verify the tool/user frame, payload, workspace, and emergency stop before continuing.")
    print(
        f"mode={settings.mode}, duration={settings.duration_s:.3f}s, "
        f"amplitude={settings.amplitude_mm:.3f}mm, period={settings.period_s * 1000:.1f}ms"
    )
    if input("Type SERVO_P to connect and continue: ").strip() != "SERVO_P":
        raise SystemExit("cancelled before connecting")

    logging.basicConfig(level=logging.INFO)
    rc = create_rc(args.ip)
    initial_powered = initial_enabled = False
    powered_by_script = enabled_by_script = False
    servo_entered = False
    samples: list[ServoSample] = []
    stream_failure: ServoPDiagnosticError | None = None
    try:
        _login(rc)
        initial_powered, initial_enabled = _controller_state(rc)
        if not initial_powered:
            if not args.power_on:
                raise RuntimeError("controller is not powered on; rerun with --power-on after safety checks")
            _payload("power_on", rc.power_on())
            powered_by_script = True
        if not initial_enabled:
            if not args.enable:
                raise RuntimeError("robot is not enabled; rerun with --enable after safety checks")
            _payload("enable_robot", rc.enable_robot())
            enabled_by_script = True

        initial_pose = _vector(
            _payload("get_actual_tcp_position", rc.get_actual_tcp_position()),
            name="actual TCP position",
        )
        targets = build_targets(initial_pose, settings)
        maximum_step_mm = max(
            (
                float(np.linalg.norm(current[:3] - previous[:3]))
                for previous, current in zip(targets, targets[1:], strict=False)
            ),
            default=0.0,
        )
        print(f"initial TCP (mm/rad): {tuple(map(float, initial_pose))}")
        print(
            f"precomputed frames={len(targets)}, max translation step={maximum_step_mm:.6f}mm "
            f"({maximum_step_mm / settings.period_s:.3f}mm/s implied)"
        )

        _set_servo_move(rc, True, robot_id=args.robot_id)
        servo_entered = True
        try:
            samples = run_stream(rc, targets, settings, robot_id=args.robot_id)
        except ServoPDiagnosticError as exc:
            samples = list(exc.samples)
            stream_failure = exc
    finally:
        if servo_entered:
            try:
                _set_servo_move(rc, False, robot_id=args.robot_id)
            except Exception:
                logger.exception("Failed to exit Servo Move during cleanup")
        if not args.leave_powered_enabled:
            if enabled_by_script:
                try:
                    _payload("disable_robot", rc.disable_robot())
                except Exception:
                    logger.exception("Failed to restore disabled state")
            if powered_by_script:
                try:
                    _payload("power_off", rc.power_off())
                except Exception:
                    logger.exception("Failed to restore powered-off state")
        logout = getattr(rc, "logout", None) or getattr(rc, "log_out", None)
        if logout is not None:
            try:
                _payload("logout", logout())
            except Exception:
                logger.exception("Failed to log out during cleanup")

    _print_report(samples, settings)
    print(
        f"host: {platform.platform()}; Python timer resolution: {time.get_clock_info('perf_counter').resolution:.9f}s"
    )
    if args.csv is not None:
        _write_csv(args.csv, samples)
        print(f"CSV written: {args.csv}")
    if stream_failure is not None:
        raise stream_failure


if __name__ == "__main__":
    main()
