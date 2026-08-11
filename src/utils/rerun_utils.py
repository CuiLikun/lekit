import contextlib
import logging
import queue
import threading
import time
import uuid

import numpy as np
import rerun as rr
import torch


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
        Maximum queued frames.  Oldest frame is dropped when full.
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

    def __init__(self, url: str = "rerun+http://127.0.0.1:9876/proxy", max_queue_size: int = 10):
        self._url = url
        self._joint_count: int | None = None
        self._position_keys: list[str] = []
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

        self._queue: queue.Queue[tuple] = queue.Queue(maxsize=max_queue_size)
        self._stopped = threading.Event()
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
                    f"overlays/{slot}/episode_label",
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
        position_views = [
            rr.blueprint.TimeSeriesView(origin="/", contents=[key], name=key.removesuffix(".pos"))
            for key in self._position_keys
        ]

        def grid(views, columns):
            return rr.blueprint.Grid(
                *views,
                grid_columns=columns,
                row_shares=[1],
                column_shares=[1] * columns,
            )

        top_row = grid(camera_views, len(camera_views)) if camera_views else None
        bottom_views = [*joint_views, *position_views]
        bottom_row = grid(bottom_views, min(4, len(bottom_views))) if bottom_views else None

        if top_row and bottom_row:
            layout = rr.blueprint.Grid(top_row, bottom_row, grid_columns=1, row_shares=[1, 1], column_shares=[1])
        elif top_row:
            layout = top_row
        elif bottom_row:
            layout = bottom_row
        else:
            return None
        return rr.blueprint.Blueprint(layout)

    @staticmethod
    def _is_position_scalar(key: str, value) -> bool:
        if not key.endswith(".pos"):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    def _enqueue_blueprint(self) -> None:
        if self._blueprint_sent or self._joint_count is None:
            return
        key = (self._joint_count, tuple(self._camera_slots), tuple(self._position_keys))
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
        ``*.pos``                          : scalar position signals such as ``joint_1.pos`` and ``gripper.pos`` (optional)
        ``task``                           : str, task instruction; overlaid on each camera image as a Rerun-native 2D label (optional)
        ``episode_number``                 : current episode number shown at image center (optional)
        ``teleop``                         : array-like (optional)
        ``policy``                         : array-like (optional)
        ``framestep``                      : int (optional). If missing, an internal increasing sequence is used.
        """
        if self._joint_count is None:
            slots = [k for k in data if k.startswith(self._IMG_PREFIX)]
            tail = lambda k: k[len(self._IMG_PREFIX) :]  # noqa
            self._camera_slots = sorted(
                slots,
                key=lambda k: (
                    next((i for i, p in enumerate(self._CAMERA_ORDER) if tail(k).startswith(p)), 999),
                    k,
                ),
            )
            self._image_keys = slots  # all present camera keys, unsorted
            state = data.get("observation.state")
            self._joint_count = len(state) if state is not None else 0
            position_keys = [
                key for key, value in data.items() if self._is_position_scalar(key, value)
            ]
            self._position_keys = sorted(
                position_keys,
                key=lambda key: (not key.startswith("joint_"), key),
            )

        # switch_record() resets the blueprint flag while retaining the inferred
        # schema, so each new recording needs another enqueue here.
        self._enqueue_blueprint()

        with contextlib.suppress(queue.Full):
            self._queue.put_nowait((self._CMD_LOG, data))

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
                    self._log_sync(item[1])
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

    def _log_sync(self, data: dict) -> None:
        frame_seq = data.get("framestep")
        frame_seq = self._next_frame_seq if frame_seq is None else int(frame_seq)

        rr.set_time("frame", sequence=frame_seq, recording=self._rec)
        self._next_frame_seq = frame_seq + 1

        task = data.get("task")
        episode_number = data.get("episode_number")
        for name in self._image_keys:
            img = data.get(name)
            if img is None:
                continue
            arr = self._to_hwc_uint8_numpy(img)
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
            if episode_number is not None:
                h, w = arr.shape[:2]
                rr.log(
                    f"overlays/{name}/episode_label",
                    rr.Points2D(
                        [(w / 2.0, h / 2.0)],
                        labels=[f"Episode {episode_number}"],
                        show_labels=True,
                        radii=[0.0],
                    ),
                    recording=self._rec,
                )

        state = data.get("observation.state")
        for i in range(self._joint_count):
            if state is None:
                break
            rr.log(f"states/{i + 1}", rr.Scalars(float(state[i])), recording=self._rec)
            if data.get("teleop") is not None:
                rr.log(f"teleop/{i + 1}", rr.Scalars(float(data["teleop"][i])), recording=self._rec)
            if data.get("policy") is not None:
                rr.log(f"policy/{i + 1}", rr.Scalars(float(data["policy"][i])), recording=self._rec)
        for key in self._position_keys:
            value = data.get(key)
            if value is not None:
                rr.log(key, rr.Scalars(float(value)), recording=self._rec)
