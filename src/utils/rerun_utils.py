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
            rr.blueprint.Spatial2DView(origin="/", contents=[slot], name=slot) for slot in self._camera_slots
        ]
        joint_views = [
            rr.blueprint.TimeSeriesView(
                origin="/",
                contents=[f"states/{i}", f"teleop/{i}", f"policy/{i}"],
                name=f"joint_{i}",
            )
            for i in range(1, self._joint_count + 1)
        ]

        def grid(views, columns):
            return rr.blueprint.Grid(
                *views,
                grid_columns=columns,
                row_shares=[1],
                column_shares=[1] * columns,
            )

        top_row = grid(camera_views, len(camera_views)) if camera_views else None
        bottom_row = grid(joint_views, min(4, len(joint_views))) if joint_views else None

        if top_row and bottom_row:
            layout = rr.blueprint.Grid(top_row, bottom_row, grid_columns=1, row_shares=[1, 1], column_shares=[1])
        elif top_row:
            layout = top_row
        elif bottom_row:
            layout = bottom_row
        else:
            return None
        return rr.blueprint.Blueprint(layout)

    def _enqueue_blueprint(self) -> None:
        if self._blueprint_sent or self._joint_count is None:
            return
        key = (self._joint_count, tuple(self._camera_slots))
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
        ``observation.state``              : array-like, used to infer joint count
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
            self._joint_count = len(data["observation.state"])
            self._enqueue_blueprint()

        try:
            self._queue.put_nowait((self._CMD_LOG, data))
        except queue.Full:
            pass  # drop the newest frame; viewer prefers liveness over backlog

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
                import traceback

                traceback.print_exc()
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

        for name in self._image_keys:
            img = data.get(name)
            if img is None:
                continue
            rr.log(name, rr.Image(self._to_hwc_uint8_numpy(img)).compress(), recording=self._rec)

        for i in range(self._joint_count):
            rr.log(f"states/{i + 1}", rr.Scalars(float(data["observation.state"][i])), recording=self._rec)
            if data.get("teleop") is not None:
                rr.log(f"teleop/{i + 1}", rr.Scalars(float(data["teleop"][i])), recording=self._rec)
            if data.get("policy") is not None:
                rr.log(f"policy/{i + 1}", rr.Scalars(float(data["policy"][i])), recording=self._rec)
