#!/usr/bin/env python

"""Exercise configured cameras at recording cadence and write an offline HTML report.

The script deliberately creates cameras directly instead of constructing ``JakaRobot``.
This keeps the camera test independent from robot power, enable, and reset behavior while
using the same camera configuration and ``read_latest`` path as the recorder.

Typical hardware run::

    uv run python -m examples.isaac_teleop_to_jaka.camera_diagnostics \
        --duration_s=60 \
        --fps=30 \
        --robot.cameras="{ hand: {type: intelrealsense, serial_number_or_name: \
        '342522070741', width: 640, height: 480, fps: 30}}"

Use ``--demo=true --duration_s=8`` to validate report generation without a camera.
"""

import base64
import csv
import hashlib
import html
import io
import json
import math
import signal
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, Literal

import numpy as np
from PIL import Image

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers "opencv")
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.configs import parser
from lekit.robots.jaka_robot import JakaRobotConfig

DEFAULT_CAMERA_SERIAL = "342522070741"
STALE_FALLBACK_MAX_AGE_MS = 2_147_483_647
REPORT_IMAGE_WIDTH = 960
STATUS_COLORS = {
    "fresh": "#16805d",
    "timeout_reused": "#c47b16",
    "error": "#c44f47",
}


def _default_robot_config() -> JakaRobotConfig:
    return JakaRobotConfig(
        id="camera_diagnostics",
        cameras={
            "hand": RealSenseCameraConfig(
                serial_number_or_name=DEFAULT_CAMERA_SERIAL,
                width=640,
                height=480,
                fps=30,
            )
        },
    )


@dataclass
class CameraDiagnosticsConfig:
    """CLI configuration for a recording-like camera soak test."""

    robot: JakaRobotConfig = field(default_factory=_default_robot_config)
    duration_s: float = 60.0
    fps: float = 30.0
    max_age_ms: int = 500
    output_dir: Path = Path("artifacts/camera_diagnostics")
    report_name: str = "report.html"
    gallery_frames: int = 5
    demo: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be positive and finite")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if self.max_age_ms <= 0:
            raise ValueError("max_age_ms must be positive")
        if not 2 <= self.gallery_frames <= 12:
            raise ValueError("gallery_frames must be between 2 and 12")
        if Path(self.report_name).name != self.report_name or not self.report_name.endswith(".html"):
            raise ValueError("report_name must be a plain .html filename")


@dataclass
class FrameSample:
    stream: str
    loop_index: int
    elapsed_s: float
    read_ms: float
    status: Literal["fresh", "timeout_reused", "error"]
    capture_timestamp_s: float | None = None
    capture_age_ms: float | None = None
    is_new_capture: bool = False
    identical_content: bool = False
    shape: str = ""
    dtype: str = ""
    digest: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class Snapshot:
    stream: str
    label: str
    elapsed_s: float
    status: str
    read_ms: float
    image: np.ndarray


@dataclass
class StreamSummary:
    stream: str
    camera_name: str
    kind: str
    configured_fps: float
    reads: int
    frames_returned: int
    fresh_reads: int
    timeout_reads: int
    errors: int
    new_captures: int
    reused_captures: int
    identical_content: int
    estimated_dropped: int
    timeout_rate: float
    error_rate: float
    reuse_rate: float
    effective_fps: float
    interval_mean_ms: float | None
    interval_median_ms: float | None
    interval_p95_ms: float | None
    interval_max_ms: float | None
    read_mean_ms: float | None
    read_p95_ms: float | None
    max_stale_age_ms: float | None
    verdict: str
    verdict_detail: str


@dataclass
class RunResult:
    cfg: CameraDiagnosticsConfig
    started_at: datetime
    elapsed_s: float
    loop_count: int
    loop_overruns: int
    samples: list[FrameSample]
    snapshots: dict[str, list[Snapshot]]
    connection_errors: dict[str, str]
    interrupted: bool


class _SyntheticCamera:
    """Small report-path fixture used only by ``--demo``."""

    def __init__(self, fps: float, width: int = 640, height: int = 480):
        self.fps = fps
        self.width = width
        self.height = height
        self.use_rgb = True
        self.use_depth = False
        self.is_connected = False
        self.latest_timestamp: float | None = None
        self._started_at = 0.0
        self._sequence = -1
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def connect(self, warmup: bool = True) -> None:
        del warmup
        self.is_connected = True
        self._started_at = time.perf_counter()
        self._update_frame(0)

    def disconnect(self) -> None:
        self.is_connected = False

    def _update_frame(self, sequence: int) -> None:
        if sequence == self._sequence:
            return
        self._sequence = sequence
        yy, xx = np.indices((self.height, self.width))
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[..., 0] = (xx // 3 + sequence * 3) % 256
        frame[..., 1] = (yy // 2 + sequence * 2) % 256
        frame[..., 2] = ((xx + yy) // 5 + sequence * 5) % 256
        marker_x = (sequence * 11) % max(1, self.width - 40)
        frame[30:80, marker_x : marker_x + 40] = (245, 205, 66)
        self._frame = frame
        self.latest_timestamp = time.perf_counter()

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        if not self.is_connected:
            raise RuntimeError("synthetic camera is disconnected")
        elapsed = time.perf_counter() - self._started_at
        sequence = int(elapsed * self.fps)
        # Freeze briefly and report a stale frame once per demo cycle.
        frozen = sequence % 91 in range(68, 86)
        if not frozen:
            self._update_frame(sequence)
        age_ms = (time.perf_counter() - (self.latest_timestamp or 0.0)) * 1000.0
        demo_limit_ms = 220 if max_age_ms < STALE_FALLBACK_MAX_AGE_MS else max_age_ms
        if age_ms > demo_limit_ms:
            raise TimeoutError(
                f"SyntheticCamera latest frame is too old: {age_ms:.1f} ms (max allowed: {demo_limit_ms} ms)."
            )
        return self._frame


class _StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def handle(self, signum: int, frame: FrameType | None) -> None:
        del signum, frame
        self.requested = True


def _camera_timestamp(camera: Any) -> float | None:
    """Read LeRobot's monotonic capture timestamp when the backend exposes it."""

    value = getattr(camera, "latest_timestamp", None)
    if value is None:
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    return timestamp if math.isfinite(timestamp) else None


def _frame_digest(frame: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(frame)
    return hashlib.blake2b(memoryview(contiguous), digest_size=8).hexdigest()


def _stream_specs(cameras: dict[str, Any]) -> list[tuple[str, str, str, Any]]:
    specs: list[tuple[str, str, str, Any]] = []
    for camera_name, camera in cameras.items():
        if getattr(camera, "use_rgb", True):
            specs.append((camera_name, camera_name, "RGB", camera.read_latest))
        if getattr(camera, "use_depth", False):
            read_depth = getattr(camera, "read_latest_depth", None)
            if read_depth is None:
                raise TypeError(f"Camera {camera_name!r} enables depth but has no read_latest_depth()")
            specs.append((f"{camera_name}_depth", camera_name, "Depth", read_depth))
    return specs


def _snapshot_label(slot: int, slots: int) -> str:
    if slot == 0:
        return "开始"
    if slot == slots - 1:
        return "结束"
    return f"进度 {slot / (slots - 1):.0%}"


def run_diagnostics(cfg: CameraDiagnosticsConfig) -> RunResult:
    """Run the camera loop and retain compact metrics plus representative frames."""

    if cfg.demo:
        cameras: dict[str, Any] = {"demo_hand": _SyntheticCamera(cfg.fps)}
    else:
        if not cfg.robot.cameras:
            raise ValueError("No cameras configured; pass --robot.cameras=...")
        cameras = make_cameras_from_configs(cfg.robot.cameras)

    started_at = datetime.now().astimezone()
    connection_errors: dict[str, str] = {}
    connected: dict[str, Any] = {}
    for name, camera in cameras.items():
        try:
            camera.connect(warmup=True)
            connected[name] = camera
            print(f"[camera] connected: {name}")
        except Exception as exc:  # Report connection failures instead of losing the artifact.
            connection_errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"[camera] connection failed: {name}: {connection_errors[name]}")

    samples: list[FrameSample] = []
    snapshots_by_slot: dict[str, dict[int, Snapshot]] = defaultdict(dict)
    timeout_snapshots: dict[str, Snapshot] = {}
    last_digest: dict[str, str] = {}
    last_capture_timestamp: dict[str, float] = {}
    stop = _StopFlag()
    previous_handlers = {sig: signal.signal(sig, stop.handle) for sig in (signal.SIGINT, signal.SIGTERM)}

    started = time.perf_counter()
    loop_count = 0
    loop_overruns = 0
    specs = _stream_specs(connected)
    target_loops = max(1, math.ceil(cfg.duration_s * cfg.fps))

    try:
        while connected and not stop.requested:
            loop_started = time.perf_counter()
            elapsed_s = loop_started - started
            if elapsed_s >= cfg.duration_s:
                break

            for stream, camera_name, _kind, reader in specs:
                camera = connected[camera_name]
                read_started = time.perf_counter()
                status: Literal["fresh", "timeout_reused", "error"] = "fresh"
                error_type = ""
                error_message = ""
                frame: np.ndarray | None = None
                try:
                    frame = reader(max_age_ms=cfg.max_age_ms)
                except TimeoutError as exc:
                    status = "timeout_reused"
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    try:
                        frame = reader(max_age_ms=STALE_FALLBACK_MAX_AGE_MS)
                    except Exception as fallback_exc:
                        status = "error"
                        error_type = type(fallback_exc).__name__
                        error_message = f"freshness timeout: {exc}; stale fallback failed: {fallback_exc}"
                except Exception as exc:
                    status = "error"
                    error_type = type(exc).__name__
                    error_message = str(exc)

                read_finished = time.perf_counter()
                read_ms = (read_finished - read_started) * 1000.0
                capture_timestamp = _camera_timestamp(camera) if frame is not None else None
                capture_age_ms = (
                    (read_finished - capture_timestamp) * 1000.0 if capture_timestamp is not None else None
                )
                digest = _frame_digest(frame) if frame is not None else ""
                previous_timestamp = last_capture_timestamp.get(stream)
                previous_digest = last_digest.get(stream)
                if capture_timestamp is not None:
                    is_new_capture = previous_timestamp is None or not math.isclose(
                        capture_timestamp, previous_timestamp, abs_tol=1e-9
                    )
                else:
                    is_new_capture = bool(digest) and digest != previous_digest
                identical_content = bool(digest and previous_digest and digest == previous_digest)

                sample = FrameSample(
                    stream=stream,
                    loop_index=loop_count,
                    elapsed_s=elapsed_s,
                    read_ms=read_ms,
                    status=status,
                    capture_timestamp_s=capture_timestamp,
                    capture_age_ms=capture_age_ms,
                    is_new_capture=is_new_capture,
                    identical_content=identical_content,
                    shape="x".join(map(str, frame.shape)) if frame is not None else "",
                    dtype=str(frame.dtype) if frame is not None else "",
                    digest=digest,
                    error_type=error_type,
                    error_message=error_message,
                )
                samples.append(sample)

                if digest:
                    last_digest[stream] = digest
                if capture_timestamp is not None:
                    last_capture_timestamp[stream] = capture_timestamp

                if frame is not None:
                    progress = min(1.0, elapsed_s / cfg.duration_s)
                    slot = min(cfg.gallery_frames - 1, round(progress * (cfg.gallery_frames - 1)))
                    snapshots_by_slot[stream].setdefault(
                        slot,
                        Snapshot(
                            stream=stream,
                            label=_snapshot_label(slot, cfg.gallery_frames),
                            elapsed_s=elapsed_s,
                            status=status,
                            read_ms=read_ms,
                            image=np.array(frame, copy=True),
                        ),
                    )
                    if status == "timeout_reused" and stream not in timeout_snapshots:
                        timeout_snapshots[stream] = Snapshot(
                            stream=stream,
                            label="首次超时后复用",
                            elapsed_s=elapsed_s,
                            status=status,
                            read_ms=read_ms,
                            image=np.array(frame, copy=True),
                        )

            loop_count += 1
            if loop_count % max(1, round(cfg.fps * 5)) == 0:
                timeout_count = sum(sample.status == "timeout_reused" for sample in samples)
                print(
                    f"[camera] {elapsed_s:6.1f}/{cfg.duration_s:.1f} s | "
                    f"loops {loop_count}/{target_loops} | timeouts {timeout_count}"
                )

            deadline = started + loop_count / cfg.fps
            remaining_s = deadline - time.perf_counter()
            if remaining_s > 0:
                time.sleep(remaining_s)
            else:
                loop_overruns += 1
    finally:
        elapsed_s = time.perf_counter() - started
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for name, camera in connected.items():
            try:
                camera.disconnect()
                print(f"[camera] disconnected: {name}")
            except Exception as exc:
                print(f"[camera] disconnect failed: {name}: {exc}")

    snapshots: dict[str, list[Snapshot]] = {}
    for stream, slots in snapshots_by_slot.items():
        ordered = [slots[index] for index in sorted(slots)]
        if stream in timeout_snapshots:
            ordered.append(timeout_snapshots[stream])
        snapshots[stream] = ordered

    return RunResult(
        cfg=cfg,
        started_at=started_at,
        elapsed_s=elapsed_s,
        loop_count=loop_count,
        loop_overruns=loop_overruns,
        samples=samples,
        snapshots=snapshots,
        connection_errors=connection_errors,
        interrupted=stop.requested,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _capture_intervals_ms(samples: list[FrameSample]) -> list[float]:
    timestamps = [
        sample.capture_timestamp_s
        for sample in samples
        if sample.is_new_capture and sample.capture_timestamp_s is not None
    ]
    if len(timestamps) < 2:
        timestamps = [sample.elapsed_s for sample in samples if sample.is_new_capture]
    return [
        (current - previous) * 1000.0
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
        if current > previous
    ]


def _estimate_drops(intervals_ms: list[float], configured_fps: float) -> int:
    expected_ms = 1000.0 / configured_fps
    # Half an expected period tolerates scheduler and camera timestamp jitter.
    return sum(max(0, round(interval_ms / expected_ms) - 1) for interval_ms in intervals_ms)


def _stream_verdict(
    *,
    frames_returned: int,
    timeout_rate: float,
    error_rate: float,
    effective_fps: float,
    configured_fps: float,
    interval_p95_ms: float | None,
    estimated_dropped: int,
    new_captures: int,
) -> tuple[str, str]:
    expected_ms = 1000.0 / configured_fps
    drop_rate = estimated_dropped / max(1, new_captures + estimated_dropped)
    if frames_returned == 0:
        return "FAIL", "没有取得任何画面"
    if error_rate > 0 or timeout_rate > 0.01 or effective_fps < configured_fps * 0.80:
        return "FAIL", "存在读取失败、持续超时或有效帧率明显不足"
    if (
        timeout_rate > 0
        or drop_rate > 0.01
        or effective_fps < configured_fps * 0.90
        or (interval_p95_ms is not None and interval_p95_ms > expected_ms * 1.5)
    ):
        return "CHECK", "可以录制，但建议先排查帧间隔尖峰或少量超时"
    return "PASS", "读取连续性满足本次测试阈值"


def summarize_streams(result: RunResult) -> list[StreamSummary]:
    configured = result.cfg.robot.cameras
    grouped: dict[str, list[FrameSample]] = defaultdict(list)
    for sample in result.samples:
        grouped[sample.stream].append(sample)

    summaries: list[StreamSummary] = []
    for stream, samples in grouped.items():
        camera_name = stream.removesuffix("_depth")
        kind = "Depth" if stream.endswith("_depth") else "RGB"
        camera_cfg = configured.get(camera_name)
        configured_fps = float(getattr(camera_cfg, "fps", None) or result.cfg.fps)
        frames_returned = sum(bool(sample.digest) for sample in samples)
        fresh_reads = sum(sample.status == "fresh" for sample in samples)
        timeout_reads = sum(sample.status == "timeout_reused" for sample in samples)
        errors = sum(sample.status == "error" for sample in samples)
        new_captures = sum(sample.is_new_capture for sample in samples)
        reused_captures = max(0, frames_returned - new_captures)
        identical_content = sum(sample.identical_content for sample in samples)
        intervals_ms = _capture_intervals_ms(samples)
        read_ms = [sample.read_ms for sample in samples]
        stale_ages = [
            sample.capture_age_ms
            for sample in samples
            if sample.status == "timeout_reused" and sample.capture_age_ms is not None
        ]
        estimated_dropped = _estimate_drops(intervals_ms, configured_fps)
        capture_span_s = sum(intervals_ms) / 1000.0
        effective_fps = (len(intervals_ms) / capture_span_s) if capture_span_s > 0 else 0.0
        timeout_rate = timeout_reads / max(1, len(samples))
        error_rate = errors / max(1, len(samples))
        reuse_rate = reused_captures / max(1, frames_returned)
        interval_p95_ms = _percentile(intervals_ms, 95)
        verdict, detail = _stream_verdict(
            frames_returned=frames_returned,
            timeout_rate=timeout_rate,
            error_rate=error_rate,
            effective_fps=effective_fps,
            configured_fps=configured_fps,
            interval_p95_ms=interval_p95_ms,
            estimated_dropped=estimated_dropped,
            new_captures=new_captures,
        )
        summaries.append(
            StreamSummary(
                stream=stream,
                camera_name=camera_name,
                kind=kind,
                configured_fps=configured_fps,
                reads=len(samples),
                frames_returned=frames_returned,
                fresh_reads=fresh_reads,
                timeout_reads=timeout_reads,
                errors=errors,
                new_captures=new_captures,
                reused_captures=reused_captures,
                identical_content=identical_content,
                estimated_dropped=estimated_dropped,
                timeout_rate=timeout_rate,
                error_rate=error_rate,
                reuse_rate=reuse_rate,
                effective_fps=effective_fps,
                interval_mean_ms=statistics.fmean(intervals_ms) if intervals_ms else None,
                interval_median_ms=statistics.median(intervals_ms) if intervals_ms else None,
                interval_p95_ms=interval_p95_ms,
                interval_max_ms=max(intervals_ms) if intervals_ms else None,
                read_mean_ms=statistics.fmean(read_ms) if read_ms else None,
                read_p95_ms=_percentile(read_ms, 95),
                max_stale_age_ms=max(stale_ages) if stale_ages else None,
                verdict=verdict,
                verdict_detail=detail,
            )
        )
    return sorted(summaries, key=lambda summary: summary.stream)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return value


def _fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}{unit}"


def _frame_to_data_uri(frame: np.ndarray) -> str:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 2:
        finite = array[np.isfinite(array)]
        if finite.size:
            low, high = np.percentile(finite, (2, 98))
            scale = max(float(high - low), 1.0)
            normalized = np.clip((array.astype(float) - low) / scale, 0.0, 1.0)
        else:
            normalized = np.zeros_like(array, dtype=float)
        red = np.clip(1.7 * normalized, 0.0, 1.0)
        green = np.clip(1.7 * (1.0 - np.abs(normalized - 0.5) * 2.0), 0.0, 1.0)
        blue = np.clip(1.7 * (1.0 - normalized), 0.0, 1.0)
        array = (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    if image.width > REPORT_IMAGE_WIDTH:
        height = round(image.height * REPORT_IMAGE_WIDTH / image.width)
        image = image.resize((REPORT_IMAGE_WIDTH, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _line_chart(samples: list[FrameSample], configured_fps: float) -> str:
    intervals = _capture_intervals_ms(samples)
    if not intervals:
        return '<div class="chart-empty">有效采集点不足，无法绘制帧间隔。</div>'
    width, height = 920, 250
    pad_left, pad_right, pad_top, pad_bottom = 54, 18, 22, 38
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    expected = 1000.0 / configured_fps
    ceiling = max(expected * 3.0, _percentile(intervals, 99) or expected)
    visible = [min(value, ceiling) for value in intervals]
    points = []
    for index, value in enumerate(visible):
        x = pad_left + index / max(1, len(visible) - 1) * plot_w
        y = pad_top + (1.0 - value / ceiling) * plot_h
        points.append(f"{x:.1f},{y:.1f}")
    expected_y = pad_top + (1.0 - expected / ceiling) * plot_h
    p95 = _percentile(intervals, 95) or 0.0
    return f"""
      <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="帧间隔折线图">
        <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}" class="axis"/>
        <line x1="{pad_left}" y1="{expected_y:.1f}" x2="{width - pad_right}" y2="{expected_y:.1f}" class="target"/>
        <polyline points="{" ".join(points)}" class="interval-line"/>
        <text x="{pad_left}" y="{expected_y - 7:.1f}" class="target-label">目标 {expected:.1f} ms</text>
        <text x="{pad_left}" y="{height - 10}" class="axis-label">开始</text>
        <text x="{width - pad_right}" y="{height - 10}" text-anchor="end" class="axis-label">结束 · {len(intervals)} 个间隔</text>
        <text x="{width - pad_right}" y="{pad_top + 2}" text-anchor="end" class="chart-note">P95 {p95:.1f} ms · 图顶 {ceiling:.1f} ms</text>
      </svg>
    """


def _status_timeline(samples: list[FrameSample]) -> str:
    if not samples:
        return '<div class="chart-empty">没有读取记录。</div>'
    max_elapsed = max(sample.elapsed_s for sample in samples) or 1.0
    bars = []
    for sample in samples:
        x = sample.elapsed_s / max_elapsed * 100.0
        color = STATUS_COLORS[sample.status]
        bars.append(
            f'<i style="left:{x:.3f}%;background:{color}" title="{sample.elapsed_s:.3f}s · {sample.status}"></i>'
        )
    return f'<div class="timeline">{"".join(bars)}</div>'


def _gallery(snapshots: list[Snapshot]) -> str:
    if not snapshots:
        return '<div class="chart-empty">本次没有取得可展示的画面。</div>'
    cards = []
    for snapshot in snapshots:
        status = html.escape(snapshot.status)
        cards.append(
            f"""
            <figure class="shot">
              <img src="{_frame_to_data_uri(snapshot.image)}" alt="{html.escape(snapshot.stream)} {html.escape(snapshot.label)}">
              <figcaption>
                <strong>{html.escape(snapshot.label)}</strong>
                <span>{snapshot.elapsed_s:.2f} s · {status} · read {snapshot.read_ms:.2f} ms</span>
              </figcaption>
            </figure>
            """
        )
    return f'<div class="gallery">{"".join(cards)}</div>'


def _verdict_class(verdict: str) -> str:
    return {"PASS": "pass", "CHECK": "check", "FAIL": "fail"}.get(verdict, "check")


def _overall_verdict(summaries: list[StreamSummary], connection_errors: dict[str, str]) -> tuple[str, str]:
    verdicts = {summary.verdict for summary in summaries}
    if connection_errors or not summaries or "FAIL" in verdicts:
        return "FAIL", "本次结果不建议直接用于正式采集，请先处理报告中的失败项。"
    if "CHECK" in verdicts:
        return "CHECK", "相机可以持续出图，但有指标需要复查，建议处理后再跑一次长测。"
    return "PASS", "本次持续读取未发现达到告警阈值的超时、失败或帧率异常。"


def _config_rows(cfg: CameraDiagnosticsConfig) -> str:
    rows = []
    cameras = cfg.robot.cameras
    if cfg.demo:
        cameras = {"demo_hand": {"type": "synthetic", "width": 640, "height": 480, "fps": cfg.fps}}
    for name, camera_cfg in cameras.items():
        data = _jsonable(asdict(camera_cfg)) if hasattr(camera_cfg, "__dataclass_fields__") else camera_cfg
        camera_type = data.get("type") or getattr(camera_cfg, "type", "unknown")
        rows.append(
            f"<tr><td><strong>{html.escape(name)}</strong></td>"
            f"<td>{html.escape(str(camera_type))}</td>"
            f"<td>{html.escape(str(data.get('serial_number_or_name', data.get('index_or_path', '—'))))}</td>"
            f"<td>{html.escape(str(data.get('width', '—')))} × {html.escape(str(data.get('height', '—')))}</td>"
            f"<td>{html.escape(str(data.get('fps', '—')))} Hz</td>"
            f"<td><code>{html.escape(json.dumps(data, ensure_ascii=False, default=str))}</code></td></tr>"
        )
    return "".join(rows)


def build_report_html(result: RunResult, summaries: list[StreamSummary]) -> str:
    verdict, verdict_text = _overall_verdict(summaries, result.connection_errors)
    verdict_css = _verdict_class(verdict)
    grouped: dict[str, list[FrameSample]] = defaultdict(list)
    for sample in result.samples:
        grouped[sample.stream].append(sample)
    total_timeouts = sum(summary.timeout_reads for summary in summaries)
    total_errors = sum(summary.errors for summary in summaries)
    total_drops = sum(summary.estimated_dropped for summary in summaries)
    generated_at = datetime.now().astimezone()

    stream_rows = (
        "".join(
            f"""
        <tr>
          <td><strong>{html.escape(summary.stream)}</strong><small>{summary.kind}</small></td>
          <td><span class="badge {_verdict_class(summary.verdict)}">{summary.verdict}</span></td>
          <td>{summary.frames_returned} / {summary.reads}</td>
          <td>{summary.timeout_reads} <small>{summary.timeout_rate:.2%}</small></td>
          <td>{summary.reused_captures} <small>{summary.reuse_rate:.2%}</small></td>
          <td>{summary.estimated_dropped}</td>
          <td>{summary.effective_fps:.2f} / {summary.configured_fps:.1f}</td>
          <td>{_fmt(summary.interval_p95_ms, " ms")}</td>
          <td>{_fmt(summary.read_p95_ms, " ms")}</td>
        </tr>
        """
            for summary in summaries
        )
        or '<tr><td colspan="9" class="empty-cell">没有可汇总的相机流</td></tr>'
    )

    connection_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>连接失败</td><td>{html.escape(message)}</td></tr>"
        for name, message in result.connection_errors.items()
    )
    sample_errors = [sample for sample in result.samples if sample.status != "fresh"]
    event_rows = connection_rows + "".join(
        f"<tr><td>{html.escape(sample.stream)}</td><td>{sample.elapsed_s:.3f} s · {html.escape(sample.status)}</td>"
        f"<td>{html.escape(sample.error_type + ': ' + sample.error_message)}</td></tr>"
        for sample in sample_errors[:100]
    )
    if not event_rows:
        event_rows = '<tr><td colspan="3" class="empty-cell">未记录到超时或读取错误</td></tr>'

    stream_sections = []
    for summary in summaries:
        stream_samples = grouped[summary.stream]
        stream_sections.append(
            f"""
            <section class="stream-section">
              <div class="section-heading">
                <div><p class="kicker">STREAM / {html.escape(summary.kind.upper())}</p><h2>{html.escape(summary.stream)}</h2></div>
                <span class="badge large {_verdict_class(summary.verdict)}">{summary.verdict}</span>
              </div>
              <p class="stream-note">{html.escape(summary.verdict_detail)}</p>
              <div class="metric-grid compact">
                <div><span>有效帧率</span><strong>{summary.effective_fps:.2f}<small> / {summary.configured_fps:.1f} Hz</small></strong></div>
                <div><span>帧间隔 P95</span><strong>{_fmt(summary.interval_p95_ms, " ms")}</strong></div>
                <div><span>帧间隔最大值</span><strong>{_fmt(summary.interval_max_ms, " ms")}</strong></div>
                <div><span>读取耗时 P95</span><strong>{_fmt(summary.read_p95_ms, " ms")}</strong></div>
                <div><span>超时后复用</span><strong>{summary.timeout_reads}<small> · {summary.timeout_rate:.2%}</small></strong></div>
                <div><span>估算丢帧</span><strong>{summary.estimated_dropped}</strong></div>
                <div><span>重复采样</span><strong>{summary.reused_captures}<small> · {summary.reuse_rate:.2%}</small></strong></div>
                <div><span>相同画面内容</span><strong>{summary.identical_content}</strong></div>
              </div>
              <div class="visual-block">
                <div class="visual-title"><strong>帧间隔</strong><span>越贴近绿色目标线越稳定；尖峰意味着新画面没有按期到达。</span></div>
                {_line_chart(stream_samples, summary.configured_fps)}
              </div>
              <div class="visual-block timeline-block">
                <div class="visual-title"><strong>读取状态时间线</strong><span><b class="dot fresh"></b>新鲜 <b class="dot timeout"></b>超时后复用 <b class="dot error"></b>失败</span></div>
                {_status_timeline(stream_samples)}
              </div>
              <div class="visual-block">
                <div class="visual-title"><strong>代表性画面</strong><span>按测试进度抽样，额外保留首次超时后复用的画面。</span></div>
                {_gallery(result.snapshots.get(summary.stream, []))}
              </div>
            </section>
            """
        )

    interrupted_note = " · 操作者提前停止" if result.interrupted else ""
    demo_note = (
        '<div class="notice demo"><strong>演示数据</strong> 本报告由 --demo 生成，只用于验证报告链路，不能代表真实相机质量。</div>'
        if result.cfg.demo
        else ""
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f3f5f2">
  <title>JAKA 录制相机持续读取测试报告</title>
  <style>
    :root {{ --paper:#f3f5f2; --panel:#fff; --ink:#17201d; --muted:#65716b; --line:#d6ded8; --deep:#173b37; --green:#16805d; --mint:#dcefe6; --amber:#a96812; --sun:#fff1d4; --red:#aa3f39; --rose:#f7dfdc; --blue:#326b91; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font:14px/1.55 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .shell {{ max-width:1240px; margin:auto; padding:24px 28px 72px; }}
    header {{ display:flex; align-items:center; justify-content:space-between; gap:24px; padding:0 0 20px; border-bottom:1px solid var(--line); }}
    .brand {{ display:flex; align-items:center; gap:12px; font-weight:800; }} .mark {{ width:38px; height:38px; display:grid; place-items:center; border-radius:6px; background:var(--deep); color:white; font:800 12px/1 ui-monospace,monospace; }}
    .brand small {{ display:block; color:var(--muted); font-size:10px; letter-spacing:.12em; }} button {{ padding:8px 11px; border:1px solid var(--line); border-radius:5px; background:white; color:var(--ink); cursor:pointer; font:700 12px inherit; }}
    .hero {{ display:grid; grid-template-columns:1.12fr .88fr; gap:42px; padding:52px 0 44px; align-items:end; }} .kicker {{ margin:0 0 7px; color:var(--green); font:800 10px/1.3 ui-monospace,monospace; letter-spacing:.14em; }}
    h1 {{ max-width:760px; margin:0; font-size:clamp(38px,5vw,64px); line-height:1.02; letter-spacing:0; }} .hero-copy {{ margin:18px 0 0; max-width:700px; color:var(--muted); font-size:16px; }}
    .verdict-card {{ padding:24px; border:1px solid var(--line); border-left:6px solid var(--green); border-radius:7px; background:white; }} .verdict-card.check {{ border-left-color:var(--amber); }} .verdict-card.fail {{ border-left-color:var(--red); }}
    .verdict-card span {{ color:var(--muted); font:700 11px ui-monospace,monospace; letter-spacing:.1em; }} .verdict-card strong {{ display:block; margin:8px 0; font-size:34px; }} .verdict-card p {{ margin:0; color:var(--muted); }}
    .notice {{ margin-bottom:24px; padding:13px 15px; border:1px solid #e1bd72; border-radius:6px; background:var(--sun); color:#704a10; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(6,1fr); border:1px solid var(--line); border-radius:7px; background:white; overflow:hidden; }} .metric-grid>div {{ min-width:0; padding:18px; border-right:1px solid var(--line); }} .metric-grid>div:last-child {{ border:0; }}
    .metric-grid span {{ display:block; color:var(--muted); font-size:11px; }} .metric-grid strong {{ display:block; margin-top:5px; font-size:24px; line-height:1.1; }} .metric-grid strong small {{ color:var(--muted); font-size:11px; font-weight:600; }}
    section {{ padding:40px 0; border-top:1px solid var(--line); }} .section-heading {{ display:flex; align-items:end; justify-content:space-between; gap:20px; margin-bottom:20px; }} h2 {{ margin:0; font-size:29px; line-height:1.1; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:7px; background:white; }} table {{ width:100%; border-collapse:collapse; }} th {{ padding:11px 13px; border-bottom:1px solid var(--line); color:var(--muted); font-size:10px; letter-spacing:.08em; text-align:left; white-space:nowrap; }} td {{ padding:13px; border-bottom:1px solid #edf0ed; vertical-align:top; }} tr:last-child td {{ border:0; }} td small {{ display:block; color:var(--muted); }} td code {{ display:block; max-width:430px; overflow-wrap:anywhere; color:#31544a; font-size:10px; }}
    .badge {{ display:inline-flex; align-items:center; min-width:58px; justify-content:center; padding:4px 8px; border-radius:4px; background:var(--mint); color:var(--green); font:800 10px ui-monospace,monospace; }} .badge.check {{ background:var(--sun); color:var(--amber); }} .badge.fail {{ background:var(--rose); color:var(--red); }} .badge.large {{ padding:8px 12px; font-size:12px; }}
    .stream-note {{ margin:-8px 0 19px; color:var(--muted); }} .metric-grid.compact {{ grid-template-columns:repeat(4,1fr); }} .metric-grid.compact>div {{ border-bottom:1px solid var(--line); }} .metric-grid.compact>div:nth-child(4n) {{ border-right:0; }} .metric-grid.compact>div:nth-last-child(-n+4) {{ border-bottom:0; }}
    .visual-block {{ margin-top:18px; padding:18px; border:1px solid var(--line); border-radius:7px; background:white; }} .visual-title {{ display:flex; justify-content:space-between; gap:20px; margin-bottom:12px; }} .visual-title span {{ color:var(--muted); font-size:11px; }}
    .chart {{ display:block; width:100%; height:auto; }} .axis {{ stroke:#bdc9c1; }} .target {{ stroke:var(--green); stroke-width:1.3; stroke-dasharray:5 5; }} .interval-line {{ fill:none; stroke:var(--blue); stroke-width:1.6; vector-effect:non-scaling-stroke; }} .axis-label,.target-label,.chart-note {{ fill:#718078; font:11px ui-monospace,monospace; }} .target-label {{ fill:var(--green); }}
    .timeline {{ position:relative; height:36px; overflow:hidden; border:1px solid var(--line); border-radius:4px; background:#edf1ee; }} .timeline i {{ position:absolute; top:0; width:max(2px,.12%); height:100%; }} .dot {{ display:inline-block; width:7px; height:7px; margin:0 3px 0 8px; border-radius:50%; }} .dot.fresh {{ background:var(--green); }} .dot.timeout {{ background:#c47b16; }} .dot.error {{ background:#c44f47; }}
    .gallery {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }} .shot {{ margin:0; overflow:hidden; border:1px solid var(--line); border-radius:5px; background:#f8faf8; }} .shot img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#e7ebe8; }} .shot figcaption {{ display:flex; justify-content:space-between; gap:10px; padding:9px 10px; }} .shot figcaption span {{ color:var(--muted); font-size:10px; text-align:right; }}
    .chart-empty,.empty-cell {{ padding:28px; color:var(--muted); text-align:center; }} .definitions {{ display:grid; grid-template-columns:repeat(2,1fr); gap:13px; }} .definition {{ padding:17px; border-left:3px solid var(--green); background:white; }} .definition strong {{ display:block; margin-bottom:4px; }} .definition p {{ margin:0; color:var(--muted); font-size:12px; }}
    .event-table td:last-child {{ max-width:800px; overflow-wrap:anywhere; font:11px/1.5 ui-monospace,monospace; }} .footer {{ display:flex; justify-content:space-between; gap:20px; padding-top:28px; color:var(--muted); font-size:11px; }}
    @media(max-width:900px) {{ .hero {{ grid-template-columns:1fr; }} .metric-grid,.metric-grid.compact {{ grid-template-columns:repeat(2,1fr); }} .metric-grid>div,.metric-grid.compact>div {{ border-right:1px solid var(--line)!important; border-bottom:1px solid var(--line)!important; }} .metric-grid>div:nth-child(2n) {{ border-right:0!important; }} .gallery {{ grid-template-columns:repeat(2,1fr); }} }}
    @media(max-width:560px) {{ .shell {{ padding:18px 15px 45px; }} header button {{ display:none; }} h1 {{ font-size:38px; }} .metric-grid,.metric-grid.compact,.definitions,.gallery {{ grid-template-columns:1fr; }} .metric-grid>div {{ border-right:0!important; }} .visual-title,.shot figcaption,.footer {{ display:block; }} .shot figcaption span {{ display:block; margin-top:3px; text-align:left; }} }}
    @media print {{ body {{ background:white; }} .shell {{ max-width:none; padding:0; }} header button {{ display:none; }} .stream-section {{ break-before:page; }} .visual-block,.table-wrap,.metric-grid {{ break-inside:avoid; }} }}
  </style>
</head>
<body>
<div class="shell">
  <header><div class="brand"><span class="mark">CAM</span><div>JAKA CAMERA DIAGNOSTICS<small>RECORDING-CADENCE SOAK TEST</small></div></div><button onclick="window.print()">打印报告</button></header>
  <main>
    <div class="hero">
      <div><p class="kicker">TEST REPORT / {generated_at:%Y-%m-%d}</p><h1>录制相机<br>持续读取测试</h1><p class="hero-copy">按 {result.cfg.fps:.1f} Hz 的真实录制节奏轮询当前相机配置，并分别统计读取超时、缓存帧复用、读取失败、估算丢帧和有效帧率。</p></div>
      <div class="verdict-card {verdict_css}"><span>OVERALL VERDICT</span><strong>{verdict}</strong><p>{html.escape(verdict_text)}</p></div>
    </div>
    {demo_note}
    <div class="metric-grid">
      <div><span>实测时长</span><strong>{result.elapsed_s:.1f}<small> s{interrupted_note}</small></strong></div>
      <div><span>循环次数</span><strong>{result.loop_count}</strong></div>
      <div><span>循环超期</span><strong>{result.loop_overruns}</strong></div>
      <div><span>新鲜度超时</span><strong>{total_timeouts}</strong></div>
      <div><span>读取失败</span><strong>{total_errors + len(result.connection_errors)}</strong></div>
      <div><span>估算丢帧</span><strong>{total_drops}</strong></div>
    </div>

    <section>
      <div class="section-heading"><div><p class="kicker">01 / SETUP</p><h2>测试条件</h2></div></div>
      <div class="table-wrap"><table><thead><tr><th>相机</th><th>类型</th><th>设备</th><th>分辨率</th><th>配置帧率</th><th>完整配置</th></tr></thead><tbody>{_config_rows(result.cfg)}</tbody></table></div>
      <div class="definitions" style="margin-top:14px">
        <div class="definition"><strong>读取路径</strong><p>先调用 read_latest(max_age_ms={result.cfg.max_age_ms})。若新鲜度检查超时，再以无限制年龄读取驱动缓存中的最新帧，和当前 record.py 的补帧方向一致。</p></div>
        <div class="definition"><strong>运行信息</strong><p>开始 {result.started_at:%Y-%m-%d %H:%M:%S %z}，报告 {generated_at:%Y-%m-%d %H:%M:%S %z}。测试只连接相机，不连接机械臂控制器。</p></div>
      </div>
    </section>

    <section>
      <div class="section-heading"><div><p class="kicker">02 / SUMMARY</p><h2>各相机流结论</h2></div></div>
      <div class="table-wrap"><table><thead><tr><th>流</th><th>判定</th><th>返回 / 读取</th><th>超时</th><th>重复采样</th><th>估算丢帧</th><th>有效 / 配置 FPS</th><th>间隔 P95</th><th>读取 P95</th></tr></thead><tbody>{stream_rows}</tbody></table></div>
    </section>

    {"".join(stream_sections)}

    <section>
      <div class="section-heading"><div><p class="kicker">EVENT LOG</p><h2>异常事件</h2></div><span>最多展示 100 条，完整数据见 samples.csv</span></div>
      <div class="table-wrap"><table class="event-table"><thead><tr><th>相机流</th><th>时刻 / 状态</th><th>异常内容</th></tr></thead><tbody>{event_rows}</tbody></table></div>
    </section>

    <section>
      <div class="section-heading"><div><p class="kicker">METHOD</p><h2>指标怎么理解</h2></div></div>
      <div class="definitions">
        <div class="definition"><strong>新鲜度超时</strong><p>驱动最新帧的年龄超过 {result.cfg.max_age_ms} ms。只要缓存里还有画面，录制可继续补齐帧数，但这段数据的视觉内容已经滞后。</p></div>
        <div class="definition"><strong>重复采样</strong><p>相邻读取拿到同一个驱动 capture timestamp。没有 timestamp 的后端才退化为图像摘要判断。静止场景的相同像素另列为“相同画面内容”。</p></div>
        <div class="definition"><strong>估算丢帧</strong><p>按相邻新 capture timestamp 的间隔除以配置帧周期，超过 1.5 个周期后估算中间缺少的帧数。它衡量采集间隔，不等同于 USB 驱动提供的硬件 dropped-frame counter。</p></div>
        <div class="definition"><strong>有效帧率</strong><p>新 capture timestamp 数量除以这些 timestamp 覆盖的时间。报告阈值为低于配置 FPS 的 90% 进入 CHECK，低于 80% 进入 FAIL。</p></div>
        <div class="definition"><strong>循环超期</strong><p>一次循环处理所有相机流后已经超过下一个 {1000 / result.cfg.fps:.1f} ms 调度点。持续超期会让录制 loop 本身跟不上目标帧率。</p></div>
        <div class="definition"><strong>建议复测</strong><p>正式采集前建议先跑 60 秒；若偶发问题难复现，再跑 10 到 30 分钟。出现超时时同步检查 USB 线材、供电、端口速率、CPU/IO 峰值和 RealSense 日志。</p></div>
      </div>
    </section>
  </main>
  <div class="footer"><span>JAKA A5 · Isaac Teleop data capture</span><span>附件：samples.csv · summary.json</span></div>
</div>
</body>
</html>
"""


def write_artifacts(result: RunResult) -> tuple[Path, Path, Path]:
    output_dir = result.cfg.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_streams(result)

    csv_path = output_dir / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(FrameSample("", 0, 0, 0, "fresh"))))
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in result.samples)

    overall_verdict, overall_detail = _overall_verdict(summaries, result.connection_errors)
    summary_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "started_at": result.started_at.isoformat(),
        "elapsed_s": result.elapsed_s,
        "loop_count": result.loop_count,
        "loop_overruns": result.loop_overruns,
        "interrupted": result.interrupted,
        "demo": result.cfg.demo,
        "overall_verdict": overall_verdict,
        "overall_detail": overall_detail,
        "connection_errors": result.connection_errors,
        "config": _jsonable(asdict(result.cfg)),
        "streams": [_jsonable(asdict(summary)) for summary in summaries],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output_dir / result.cfg.report_name
    report_path.write_text(build_report_html(result, summaries), encoding="utf-8")
    return report_path, csv_path, summary_path


@parser.wrap()
def main(cfg: CameraDiagnosticsConfig) -> None:
    print(
        f"[camera] starting {'demo' if cfg.demo else 'hardware'} test: "
        f"{cfg.duration_s:.1f} s at {cfg.fps:.1f} Hz"
    )
    result = run_diagnostics(cfg)
    report_path, csv_path, summary_path = write_artifacts(result)
    summaries = summarize_streams(result)
    verdict, detail = _overall_verdict(summaries, result.connection_errors)
    print(f"[camera] verdict: {verdict} — {detail}")
    print(f"[camera] report:  {report_path}")
    print(f"[camera] samples: {csv_path}")
    print(f"[camera] summary: {summary_path}")


if __name__ == "__main__":
    main()
