import queue
import threading
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
    _CMD_TIME_SEQ = 1
    _CMD_BLUEPRINT = 2
    _CMD_STOP = 3
    _CMD_NEW_RECORDING = 4

    def __init__(self, url: str, max_queue_size: int = 10, y_range: tuple[float, float] | None = None):
        self._url = url
        self._y_range = y_range  # None means auto-scale
        self._joint_count: int | None = None
        self._blueprint_sent = False
        self._next_frame_seq = 0
        # Camera slots are inferred from available observation.images.* keys.
        self._camera_slots: list[str] = []

        # Isolated recording stream - never touches global rr.init() state.
        self._rec = rr.RecordingStream(
            application_id="lerobot",
            recording_id=uuid.uuid4(),
            make_default=False,
            make_thread_default=False,
        )
        self._connect_stream()

        self._queue: queue.Queue[tuple] = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="RerunLogger")
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _enqueue_blueprint(self) -> None:
        if self._blueprint_sent or self._joint_count is None:
            return

        camera_views = [
            rr.blueprint.Spatial2DView(origin="/", contents=[slot], name=slot) for slot in self._camera_slots
        ]
        joint_views = []
        for joint_no in range(1, self._joint_count + 1):
            joint_views.append(
                rr.blueprint.TimeSeriesView(
                    origin="/",
                    contents=[f"states/{joint_no}", f"teleop/{joint_no}", f"policy/{joint_no}"],
                    name=f"joint_{joint_no}",
                    axis_y=rr.blueprint.ScalarAxis(range=self._y_range) if self._y_range else None,
                )
            )

        top_row = None
        if camera_views:
            top_row = rr.blueprint.Grid(
                *camera_views,
                grid_columns=len(camera_views),
                row_shares=[1],
                column_shares=[1] * len(camera_views),
            )

        bottom_row = None
        if joint_views:
            bottom_row = rr.blueprint.Grid(
                *joint_views,
                grid_columns=min(4, len(joint_views)),
                row_shares=[1],
                column_shares=[1] * min(4, len(joint_views)),
            )

        if top_row and bottom_row:
            layout = rr.blueprint.Grid(
                top_row,
                bottom_row,
                grid_columns=1,
                row_shares=[1, 1],
                column_shares=[1],
            )
        elif top_row:
            layout = top_row
        elif bottom_row:
            layout = bottom_row
        else:
            return

        blueprint = rr.blueprint.Blueprint(layout)
        self._queue.put((self._CMD_BLUEPRINT, blueprint))
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
            image_keys = sorted(k for k in data if str(k).startswith("observation.images."))
            # Sort with priority order: left, head, right
            priority_order = {"left": 0, "head": 1, "right": 2}

            def get_priority(k):
                key_part = k.split(".")[-1]
                for priority_key, priority_value in priority_order.items():
                    if priority_key in key_part:
                        return priority_value
                return 999

            self._camera_slots = sorted(image_keys, key=lambda k: (get_priority(k), k))
            self._joint_count = len(data["observation.state"])
            self._enqueue_blueprint()

        try:
            self._queue.put_nowait((self._CMD_LOG, data))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait((self._CMD_LOG, data))

    def switch_record(self) -> None:
        """Switch to a new recording_id (e.g., when starting a new episode).

        This creates a new RecordingStream instance in the worker thread,
        allowing each episode to appear as a separate recording in the viewer.
        Also resets the blueprint so it is re-sent for the new recording.
        """
        self._blueprint_sent = False
        self._queue.put((self._CMD_NEW_RECORDING,))

    def flush(self) -> None:
        """Block until all queued items have been sent to the network."""
        self._queue.join()

    def stop(self) -> None:
        """Flush, stop the worker thread, and disconnect."""
        self.flush()
        self._queue.put((self._CMD_STOP,))
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
                cmd = item[0]
                if cmd == self._CMD_STOP:
                    return
                elif cmd == self._CMD_LOG:
                    self._log_sync(item[1])
                elif cmd == self._CMD_BLUEPRINT:
                    rr.send_blueprint(item[1], recording=self._rec)
                elif cmd == self._CMD_NEW_RECORDING:
                    # Disconnect old recording and create a new one with fresh recording_id
                    rr.disconnect(recording=self._rec)
                    self._rec = rr.RecordingStream(
                        application_id="lerobot",
                        recording_id=uuid.uuid4(),
                        make_default=False,
                        make_thread_default=False,
                    )
                    self._connect_stream()
                    self._next_frame_seq = 0
            finally:
                self._queue.task_done()

    def _to_hwc_uint8_numpy(self, image: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] <= 4:
                if image.dtype == torch.uint8:
                    return image.permute(1, 2, 0).cpu().numpy()
                return (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return image.cpu().numpy()

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                scale = 255.0 if arr.max(initial=0.0) <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _log_sync(self, data: dict) -> None:
        frame_seq = data.get("framestep")
        frame_seq = self._next_frame_seq if frame_seq is None else int(frame_seq)

        rr.set_time("frame", sequence=frame_seq, recording=self._rec)
        self._next_frame_seq = frame_seq + 1

        image_keys = [k for k in data if str(k).startswith("observation.images.")]
        images = {k: self._to_hwc_uint8_numpy(data[k]) for k in image_keys if data.get(k) is not None}
        # Log up to 3 available camera streams; do not pad missing cameras.
        for name, image in images.items():
            rr.log(name, rr.Image(image).compress(), recording=self._rec)

        for i in range(self._joint_count):
            rr.log(f"states/{i + 1}", rr.Scalars(float(data["observation.state"][i])), recording=self._rec)
            if data.get("teleop") is not None:
                rr.log(f"teleop/{i + 1}", rr.Scalars(float(data["teleop"][i])), recording=self._rec)
            if data.get("policy") is not None:
                rr.log(f"policy/{i + 1}", rr.Scalars(float(data["policy"][i])), recording=self._rec)


if __name__ == "__main__":
    import argparse
    import time

    from lerobot.replay_bot import ReplayBot, ReplayBotConfig

    parser = argparse.ArgumentParser(description="Replay dataset episode(s) and stream them to Rerun")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id, e.g. lerobot/pusht")
    parser.add_argument("--root", default=None, help="Optional local dataset root")
    parser.add_argument("--episode", default="0,1,2", help="Episode index or comma-separated indices")
    parser.add_argument(
        "--url",
        required=True,
        help="Rerun URL. Use 127.0.0.1 or localhost to spawn a local viewer, or a remote address to connect remotely.",
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Sleep seconds between frames")
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=[-3.14, 3.14],
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits for joint time series, e.g. --y-range -3.14 3.14",
    )
    parser.add_argument(
        "--no-ylim",
        action="store_true",
        help="Auto-scale Y axis instead of fixed limits.",
    )
    args = parser.parse_args()

    y_range = None if args.no_ylim else tuple(args.y_range)

    robot_cfg = ReplayBotConfig(repo_id=args.repo_id, root=args.root, episode=args.episode)
    robot = ReplayBot(robot_cfg)

    sent_frames = 0
    with RerunLogger(url=args.url, y_range=y_range) as logger:
        for episode_index in robot.episodes:
            robot.load_episode(episode_index)
            logger.switch_record()  # Switch to new recording for this episode
            while not robot.is_episode_done:
                observation = robot.get_observation()
                action = robot.get_teleop_action()
                logger.log({"framestep": robot.frame_index, **observation, **action})
                robot.send_action({})
                sent_frames += 1
                time.sleep(args.dt)

    print(f"Done - sent {sent_frames} frames to {args.url}")
