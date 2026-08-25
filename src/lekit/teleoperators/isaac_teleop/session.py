"""Shared CloudXR and Isaac Teleop session lifecycle."""

from __future__ import annotations

import abc
import contextlib
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.import_utils import is_package_available

from .config import IsaacTeleopConfig

logger = logging.getLogger(__name__)
_isaacteleop_available = is_package_available("isaacteleop")
_FORM_FACTOR_UNAVAILABLE = re.compile(r"(?:^|:\s*)-35\s*$")


@contextlib.contextmanager
def _silence_native_output(enabled: bool):
    """Temporarily silence native stdout/stderr writes during expected retries."""

    if not enabled:
        yield
        return

    null_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(stdout_fd)
        os.close(null_fd)


if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.cloudxr import CloudXRLauncher
    from isaacteleop.retargeting_engine.interface import ExecutionEvents, ExecutionState, OutputCombiner
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig
else:
    CloudXRLauncher = Any
    ExecutionEvents = Any
    ExecutionState = Any
    OutputCombiner = Any
    TeleopSession = Any
    TeleopSessionConfig = Any


def require_isaacteleop() -> None:
    if not _isaacteleop_available:
        raise ImportError(
            "Isaac XR teleoperation requires the optional dependency. Install with `uv sync --extra teleop`."
        )


class IsaacTeleopSession(Teleoperator, abc.ABC):
    """A small interface hiding CloudXR and session lifecycle details."""

    config_class = IsaacTeleopConfig

    def __init__(self, config: IsaacTeleopConfig):
        require_isaacteleop()
        super().__init__(config)
        self.config = config
        self._session: TeleopSession | None = None
        self._cloudxr_launcher: CloudXRLauncher | None = None
        self._connect_wait_callback: Callable[[], None] | None = None

    @abc.abstractmethod
    def _build_pipeline(self) -> OutputCombiner:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def send_feedback(self, _feedback: dict[str, Any]) -> None:
        """Accept standard LeRobot feedback without coupling it into control."""

    def set_connect_wait_callback(self, callback: Callable[[], None] | None) -> None:
        """Set an optional heartbeat called while waiting for an XR headset."""

        self._connect_wait_callback = callback

    def connect(self, calibrate: bool = True) -> None:
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")
        if calibrate:
            self.calibrate()
        self._ensure_cloudxr_runtime()
        try:
            config = TeleopSessionConfig(app_name=self.config.app_name, pipeline=self._build_pipeline())
            self._session = self._wait_for_xr_system(config)
        except BaseException:
            self._session = None
            self._stop_cloudxr_runtime()
            raise
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def _wait_for_xr_system(self, config: TeleopSessionConfig) -> TeleopSession:
        """Open a session, waiting while CloudXR has no headset form factor."""

        deadline = (
            None
            if self.config.connect_timeout_s is None
            else time.monotonic() + self.config.connect_timeout_s
        )
        waiting_reported = False
        while True:
            try:
                with _silence_native_output(waiting_reported):
                    session = TeleopSession(config)
                    session.__enter__()
            except RuntimeError as error:
                if not self._is_form_factor_unavailable(error):
                    raise
                if self._cloudxr_launcher is not None:
                    self._cloudxr_launcher.health_check()
                if not waiting_reported:
                    print("Waiting for an XR headset to connect... (Ctrl-C to abort)", flush=True)
                    waiting_reported = True
                callback = getattr(self, "_connect_wait_callback", None)
                if callback is not None:
                    callback()
                sleep_s = self.config.connect_retry_interval_s
                if deadline is not None:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0.0:
                        raise TimeoutError(
                            f"No XR headset became available within {self.config.connect_timeout_s:g} seconds"
                        ) from error
                    sleep_s = min(sleep_s, remaining_s)
                time.sleep(sleep_s)
            else:
                return session

    @staticmethod
    def _is_form_factor_unavailable(error: RuntimeError) -> bool:
        result = getattr(error, "result", None)
        return result == -35 or _FORM_FACTOR_UNAVAILABLE.search(str(error)) is not None

    def disconnect(self) -> None:
        try:
            if self._session is not None:
                session = self._session
                self._session = None
                session.__exit__(None, None, None)
                logger.info("Isaac Teleop session ended")
        finally:
            self._stop_cloudxr_runtime()

    def _ensure_cloudxr_runtime(self) -> None:
        if self._cloudxr_launcher is not None:
            return
        if os.environ.get("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "").strip() == "1":
            logger.info("Skipping CloudXR launch because LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1")
            return
        if not self.config.auto_launch_cloudxr:
            return
        self._cloudxr_launcher = CloudXRLauncher(
            install_dir=str(Path(self.config.cloudxr_install_dir).expanduser().resolve()),
            env_config=self.config.cloudxr_env_file,
            accept_eula=False,
        )

    def _stop_cloudxr_runtime(self) -> None:
        if self._cloudxr_launcher is None:
            return
        try:
            self._cloudxr_launcher.stop()
        except RuntimeError:
            logger.warning("CloudXR runtime could not be terminated; launcher will retry at process exit")
        else:
            self._cloudxr_launcher = None

    def _step(self) -> Any:
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")
        result = self._session.step(
            execution_events=ExecutionEvents(execution_state=ExecutionState.RUNNING, reset=False)
        )
        info = self._session.last_step_info
        if info is not None:
            if info.worker_exception is not None:
                raise RuntimeError(
                    "Isaac Teleop retargeting worker raised an exception"
                ) from info.worker_exception
            if info.frame_deadline_miss:
                logger.warning("Isaac Teleop frame deadline miss (age=%s)", info.returned_age_frames)
        return result
