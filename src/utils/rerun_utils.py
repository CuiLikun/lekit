import logging
import queue
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
import rerun as rr
import torch


class _LatestLogQueue(queue.Queue):
    """FIFO command queue that coalesces pending live-view frames."""

    def put_latest(self, item: tuple, log_command: int) -> tuple[bool, int]:
        """Replace queued log frames and report ``(accepted, dropped)``."""

        with self.not_full:
            pending = list(self.queue)
            retained = [
                queued
                for queued in pending
                if not isinstance(queued, tuple) or not queued or queued[0] != log_command
            ]
            dropped = len(pending) - len(retained)
            if dropped:
                self.queue.clear()
                self.queue.extend(retained)
                self.unfinished_tasks -= dropped
                if self.unfinished_tasks == 0:
                    self.all_tasks_done.notify_all()
                self.not_full.notify_all()

            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                return False, dropped

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            return True, dropped


class RerunLogger:
    """Non-blocking logger that streams robot data to local or remote Rerun viewer.

    Parameters
    ----------
    url: str
        Rerun endpoint URL. Determines where data is streamed:

        - ``rerun+http://127.0.0.1:9876/proxy`` — spawn & connect to a local viewer
        - ``rerun+http://localhost:9876/proxy``  — same as above (localhost alias)
        - ``rerun+http://192.168.1.10:9876/proxy`` — connect to a remote viewer over gRPC
    max_queue_size: int
        Maximum queued commands. At most one pending live-view frame is retained.
    """

    _CMD_LOG = 0
    _CMD_BLUEPRINT = 1
    _CMD_NEW_RECORDING = 2

    # Sentinel object so stop() can break the worker out even if the queue is
    # blocked on a slow gRPC send (distinct from int command tags).
    _SENTINEL_STOP = object()

    # Display order for camera slots. Matched by suffix so "forehead" doesn't
    # collide with "head". Anything not in the table falls to the end.
    _CAMERA_ORDER = ("head", "left", "right")
    _IMG_PREFIX = "observation.images."
    _CURVE_PREFIXES = ("observation.", "action.", "metrics.")
    _EPISODE_FONT_SCALE = 1.0
    _EPISODE_FONT_THICKNESS = 1
    _EPISODE_TOP_MARGIN_PX = 12
    _STATUS_FONT_SCALE = 1.0
    _STATUS_FONT_THICKNESS = 1
    _STATUS_LINE_GAP_PX = 10

    def __init__(self, url: str = "rerun+http://127.0.0.1:9876/proxy", max_queue_size: int = 10):
        self._url = url
        self._joint_count: int | None = None
        self._curve_keys: list[str] = []
        self._blueprint_sent = False
        self._next_frame_seq = 0
        # Camera slots are inferred from available observation.images.* keys.
        self._camera_slots: list[str] = []
        self._image_keys: list[str] = []  # cached on first log()
        self._cached_blueprint: rr.blueprint.Blueprint | None = None
        self._cached_blueprint_key: tuple = ()

        # Isolated recording stream - never touches global rr.init() state.
        self._rec = rr.RecordingStream(
            application_id="lerobot",
            recording_id=uuid.uuid4(),
            make_default=False,
            make_thread_default=False,
        )
        self._connect_stream()

        self._queue: _LatestLogQueue = _LatestLogQueue(maxsize=max_queue_size)
        self._stopped = threading.Event()
        self._init_status()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="RerunLogger")
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_blueprint(self) -> rr.blueprint.Blueprint:
        camera_views = [
            rr.blueprint.Spatial2DView(
                origin="/",
                contents=[
                    slot,
                    f"overlays/{slot}/task_label",
                ],
                name=slot,
            )
            for slot in self._camera_slots
        ]
        joint_views = [
            rr.blueprint.TimeSeriesView(
                origin="/",
                contents=[f"states/{i}", f"teleop/{i}", f"policy/{i}"],
                name=f"joint_{i}",
            )
            for i in range(1, self._joint_count + 1)
        ]
        curve_groups: dict[str, list[str]] = {}
        for key in self._curve_keys:
            curve_groups.setdefault(self._curve_group_name(key), []).append(key)
        curve_views = [
            rr.blueprint.TimeSeriesView(origin="/", contents=keys, name=name)
            for name, keys in curve_groups.items()
        ]

        def grid(views, columns):
            return rr.blueprint.Grid(
                *views,
                grid_columns=columns,
                row_shares=[1],
                column_shares=[1] * columns,
            )

        top_row = grid(camera_views, len(camera_views)) if camera_views else None
        bottom_views = [*joint_views, *curve_views]
        bottom_row = grid(bottom_views, min(4, len(bottom_views))) if bottom_views else None

        if top_row and bottom_row:
            layout = rr.blueprint.Grid(
                top_row, bottom_row, grid_columns=1, row_shares=[1, 1], column_shares=[1]
            )
        elif top_row:
            layout = top_row
        elif bottom_row:
            layout = bottom_row
        else:
            return None
        return rr.blueprint.Blueprint(layout)

    @staticmethod
    def _is_curve_scalar(key: str, value) -> bool:
        if key.startswith(RerunLogger._IMG_PREFIX):
            return False
        if not key.endswith(".pos") and not key.startswith(RerunLogger._CURVE_PREFIXES):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _curve_group_name(key: str) -> str:
        signal = key
        for prefix in RerunLogger._CURVE_PREFIXES:
            if signal.startswith(prefix):
                signal = signal[len(prefix) :]
                break
        if signal in ("ee.x", "ee.y", "ee.z"):
            return "tcp_position"
        if signal in ("ee.roll", "ee.pitch", "ee.yaw"):
            return "tcp_orientation"
        return signal.removesuffix(".pos")

    def _update_schema(self, data: dict) -> None:
        previous = (
            self._joint_count,
            tuple(self._camera_slots),
            tuple(self._image_keys),
            tuple(self._curve_keys),
        )

        image_keys = set(self._image_keys)
        image_keys.update(key for key in data if key.startswith(self._IMG_PREFIX))
        tail = lambda key: key[len(self._IMG_PREFIX) :]  # noqa: E731
        self._camera_slots = sorted(
            image_keys,
            key=lambda key: (
                next((i for i, name in enumerate(self._CAMERA_ORDER) if tail(key).startswith(name)), 999),
                key,
            ),
        )
        self._image_keys = sorted(image_keys)

        state = data.get("observation.state")
        if self._joint_count is None:
            self._joint_count = len(state) if state is not None else 0
        elif state is not None:
            self._joint_count = max(self._joint_count, len(state))

        curve_keys = set(self._curve_keys)
        curve_keys.update(key for key, value in data.items() if self._is_curve_scalar(key, value))
        self._curve_keys = sorted(
            curve_keys,
            key=lambda key: (
                self._curve_group_name(key),
                not key.startswith("observation."),
                key,
            ),
        )

        current = (
            self._joint_count,
            tuple(self._camera_slots),
            tuple(self._image_keys),
            tuple(self._curve_keys),
        )
        if current != previous:
            self._blueprint_sent = False

    def _enqueue_blueprint(self) -> None:
        if self._blueprint_sent or self._joint_count is None:
            return
        key = (self._joint_count, tuple(self._camera_slots), tuple(self._curve_keys))
        if self._cached_blueprint_key != key:
            self._cached_blueprint = self._build_blueprint()
            self._cached_blueprint_key = key
        if self._cached_blueprint is None:
            return
        self._queue.put((self._CMD_BLUEPRINT, self._cached_blueprint))
        self._blueprint_sent = True

    def log(self, data: dict) -> None:
        """Enqueue a data dict for async logging (non-blocking).

        Expected keys
        -------------
        ``observation.images.*``           : np.ndarray HWC uint8, dynamic camera count (1..3)
        ``observation.state``              : array-like, used to infer joint count (optional)
        ``observation.*`` / ``action.*``   : finite scalar feedback and action curves (optional)
        ``metrics.*``                      : finite scalar runtime metrics (optional)
        ``*.pos``                          : legacy direct position signals (optional)
        ``task``                           : str, task instruction; overlaid on each camera image as a Rerun-native 2D label (optional)
        ``episode_number``                 : current episode number shown at image center (optional)
        ``record_state``                   : ``recording``/``pause`` status shown below the episode number (optional)
        ``teleop``                         : array-like (optional)
        ``policy``                         : array-like (optional)
        ``framestep``                      : int (optional). If missing, an internal increasing sequence is used.
        """
        self._update_schema(data)

        # switch_record() resets the blueprint flag while retaining the inferred
        # schema, so each new recording needs another enqueue here.
        self._enqueue_blueprint()

        accepted, coalesced = self._queue.put_latest((self._CMD_LOG, data), self._CMD_LOG)
        with self._status_lock:
            self._enqueued_frames += 1
            self._dropped_frames += coalesced + (not accepted)

    def get_status(self) -> dict[str, int | float | bool]:
        """Return a lightweight snapshot of the asynchronous logging pipeline."""

        now = time.monotonic()
        with self._status_lock:
            while self._processed_at and self._processed_at[0] < now - 1.0:
                self._processed_at.popleft()
            window_s = min(max(now - self._status_started_at, 1e-6), 1.0)
            in_flight_ms = (
                (now - self._log_started_at) * 1000.0
                if self._log_started_at is not None
                else 0.0
            )
            status = {
                "enqueued_frames": self._enqueued_frames,
                "processed_frames": self._processed_frames,
                "dropped_frames": self._dropped_frames,
                "worker_rate_hz": len(self._processed_at) / window_s,
                "log_ms": self._last_log_ms,
                "image_ms": self._last_image_ms,
                "curve_ms": self._last_curve_ms,
                "in_flight": self._log_started_at is not None,
                "in_flight_ms": in_flight_ms,
                "errors": self._errors,
            }
        status["queue_depth"] = self._queue.qsize()
        status["worker_alive"] = self._thread.is_alive()
        return status

    def switch_record(self) -> None:
        """Switch to a new recording_id (e.g., when starting a new episode).

        This creates a new RecordingStream instance in the worker thread,
        allowing each episode to appear as a separate recording in the viewer.
        Also resets the blueprint so it is re-sent for the new recording.
        """
        self._blueprint_sent = False
        self._queue.put((self._CMD_NEW_RECORDING,))

    def flush(self, timeout: float | None = None) -> None:
        """Block until all queued items have been sent to the network.

        A ``timeout`` (seconds) caps the wait; on timeout, the worker is still
        alive and any remaining items may be sent in the background.
        """
        if timeout is None:
            self._queue.join()
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return
            time.sleep(0.005)

    def stop(self) -> None:
        """Stop the worker thread and disconnect."""
        if self._stopped.is_set():
            return
        self._stopped.set()
        # SENTINEL_STOP sits behind any pending log items, so the worker drains
        # them naturally before exiting. No explicit flush() needed.
        self._queue.put(self._SENTINEL_STOP)
        self._thread.join(timeout=3.0)
        rr.disconnect(recording=self._rec)

    def __enter__(self) -> "RerunLogger":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_status(self) -> None:
        self._status_lock = threading.Lock()
        self._status_started_at = time.monotonic()
        self._enqueued_frames = 0
        self._processed_frames = 0
        self._dropped_frames = 0
        self._errors = 0
        self._last_log_ms = 0.0
        self._last_image_ms = 0.0
        self._last_curve_ms = 0.0
        self._log_started_at: float | None = None
        self._processed_at: deque[float] = deque()

    def _is_local(self) -> bool:
        return self._url and ("127.0.0.1" in self._url or "localhost" in self._url)

    def _connect_stream(self) -> None:
        if self._is_local():
            rr.spawn(recording=self._rec)
            return
        rr.connect_grpc(self._url, recording=self._rec)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL_STOP:
                    return
                cmd = item[0]
                if cmd == self._CMD_LOG:
                    started_at = time.monotonic()
                    timings = None
                    with self._status_lock:
                        self._log_started_at = started_at
                    try:
                        timings = self._log_sync(item[1])
                    finally:
                        finished_at = time.monotonic()
                        with self._status_lock:
                            self._processed_frames += 1
                            self._last_log_ms = (finished_at - started_at) * 1000.0
                            if timings is not None:
                                self._last_image_ms = timings["image_ms"]
                                self._last_curve_ms = timings["curve_ms"]
                            self._log_started_at = None
                            self._processed_at.append(finished_at)
                            while self._processed_at[0] < finished_at - 1.0:
                                self._processed_at.popleft()
                elif cmd == self._CMD_BLUEPRINT:
                    rr.send_blueprint(item[1], recording=self._rec)
                elif cmd == self._CMD_NEW_RECORDING:
                    rr.disconnect(recording=self._rec)
                    self._rec = rr.RecordingStream(
                        application_id="lerobot",
                        recording_id=uuid.uuid4(),
                        make_default=False,
                        make_thread_default=False,
                    )
                    self._connect_stream()
                    self._next_frame_seq = 0
            except Exception:
                # Rerun is auxiliary to control and recording. Keep worker failures
                # out of stdout/stderr so they cannot corrupt a live Rich panel.
                if item is not self._SENTINEL_STOP and item[0] == self._CMD_LOG:
                    with self._status_lock:
                        self._errors += 1
                logging.getLogger(__name__).debug("Rerun logging failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _to_hwc_uint8_numpy(self, image) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                image = image.permute(1, 2, 0)
            if image.dtype == torch.uint8:
                return image.cpu().numpy()
            if image.dtype.is_floating_point:
                return np.clip(image.float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            return np.clip(image.cpu().numpy(), 0, 255).astype(np.uint8)

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype == np.uint8:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _draw_episode_label(
        self,
        image: np.ndarray,
        episode_number,
        record_state: str | None = None,
    ) -> np.ndarray:
        """Draw the episode number and recorder state on a Rerun-only image copy."""

        output = np.ascontiguousarray(image.copy())
        height, width = output.shape[:2]
        text = f"Episode {episode_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._EPISODE_FONT_SCALE
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            font,
            scale,
            self._EPISODE_FONT_THICKNESS,
        )
        available_width = max(width - 2 * self._EPISODE_TOP_MARGIN_PX, 1)
        if text_width > available_width:
            scale *= available_width / text_width
            (text_width, text_height), _ = cv2.getTextSize(
                text,
                font,
                scale,
                self._EPISODE_FONT_THICKNESS,
            )

        origin = (
            max((width - text_width) // 2, 0),
            min(self._EPISODE_TOP_MARGIN_PX + text_height, height - 1),
        )
        color = (255, 255, 255) if output.ndim == 3 else 255
        outline = (0, 0, 0) if output.ndim == 3 else 0
        cv2.putText(
            output,
            text,
            origin,
            font,
            scale,
            outline,
            self._EPISODE_FONT_THICKNESS + 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            origin,
            font,
            scale,
            color,
            self._EPISODE_FONT_THICKNESS,
            cv2.LINE_AA,
        )

        if record_state:
            status_text = str(record_state).upper()
            status_scale = self._STATUS_FONT_SCALE
            (status_width, status_height), _ = cv2.getTextSize(
                status_text,
                font,
                status_scale,
                self._STATUS_FONT_THICKNESS,
            )
            if status_width > available_width:
                status_scale *= available_width / status_width
                (status_width, status_height), _ = cv2.getTextSize(
                    status_text,
                    font,
                    status_scale,
                    self._STATUS_FONT_THICKNESS,
                )
            status_origin = (
                max((width - status_width) // 2, 0),
                min(origin[1] + self._STATUS_LINE_GAP_PX + status_height, height - 1),
            )
            if output.ndim == 3:
                status_color = (255, 210, 64) if status_text == "PAUSE" else (92, 230, 140)
            else:
                status_color = 255
            cv2.putText(
                output,
                status_text,
                status_origin,
                font,
                status_scale,
                outline,
                self._STATUS_FONT_THICKNESS + 4,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                status_text,
                status_origin,
                font,
                status_scale,
                status_color,
                self._STATUS_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        return output

    def _log_sync(self, data: dict) -> dict[str, float]:
        frame_seq = data.get("framestep")
        frame_seq = self._next_frame_seq if frame_seq is None else int(frame_seq)

        rr.set_time("frame", sequence=frame_seq, recording=self._rec)
        self._next_frame_seq = frame_seq + 1

        task = data.get("task")
        episode_number = data.get("episode_number")
        record_state = data.get("record_state")
        image_started_at = time.perf_counter()
        for name in self._image_keys:
            img = data.get(name)
            if img is None:
                continue
            arr = self._to_hwc_uint8_numpy(img)
            if episode_number is not None:
                arr = self._draw_episode_label(arr, episode_number, record_state)
            rr.log(name, rr.Image(arr).compress(), recording=self._rec)
            # Overlay the task string at the bottom-center of the image via a
            # Rerun-native 2D archetype. The Spatial2DView pulls this entity
            # into the same panel, so the label floats over the image.
            if task:
                h, w = arr.shape[:2]
                rr.log(
                    f"overlays/{name}/task_label",
                    rr.Points2D(
                        [(w / 2.0, h - 24)],
                        labels=[task],
                        show_labels=True,
                        radii=[0.0],
                    ),
                    recording=self._rec,
                )
        image_ms = (time.perf_counter() - image_started_at) * 1000.0
        curve_started_at = time.perf_counter()
        state = data.get("observation.state")
        for i in range(self._joint_count):
            if state is None:
                break
            rr.log(f"states/{i + 1}", rr.Scalars(float(state[i])), recording=self._rec)
            if data.get("teleop") is not None:
                rr.log(f"teleop/{i + 1}", rr.Scalars(float(data["teleop"][i])), recording=self._rec)
            if data.get("policy") is not None:
                rr.log(f"policy/{i + 1}", rr.Scalars(float(data["policy"][i])), recording=self._rec)
        for key in self._curve_keys:
            value = data.get(key)
            if value is not None:
                rr.log(key, rr.Scalars(float(value)), recording=self._rec)
        return {
            "image_ms": image_ms,
            "curve_ms": (time.perf_counter() - curve_started_at) * 1000.0,
        }
