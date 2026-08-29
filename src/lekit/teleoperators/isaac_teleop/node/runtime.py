"""Run Isaac XR as an independent latest-frame service with a Web monitor."""

from __future__ import annotations

import argparse
import contextlib
import math
import signal
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import uvicorn

from lekit.control.controller import ControllerNode, ControllerNodeConfig
from lekit.control.model import NodePresentation
from lekit.control.runtime import Runtime

from ..config import IsaacTeleopConfig
from ..engage_authority import EngageAuthority
from ..protocol import ACTION_SCHEMA, ACTION_SCHEMA_VERSION, TeleopFrame, encode_action_frame
from ..transport import ZmqTeleopPublisher
from ..xr_controller import IsaacXRController
from .monitor import TeleopNodeState, create_monitor_app

_CLIENT_URL = "https://nvidia.github.io/IsaacTeleop/client"


class _StopRequestedError(Exception):
    """Unwind a blocking XR connection attempt after a service stop request."""


@dataclass(kw_only=True)
class TeleopNodeConfig:
    """Runtime configuration for the independent Isaac teleop service."""

    controller: IsaacTeleopConfig = field(default_factory=IsaacTeleopConfig)
    publish_endpoint: str = "tcp://127.0.0.1:5557"
    rate_hz: float = 60.0
    retry_delay_s: float = 2.0
    status_interval_s: float = 1.0
    monitor_enabled: bool = True
    monitor_host: str = "127.0.0.1"
    monitor_port: int = 8000

    def __post_init__(self) -> None:
        parsed = urlsplit(self.publish_endpoint)
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError(f"publish_endpoint has an invalid port: {self.publish_endpoint!r}") from error
        if parsed.scheme != "tcp" or not parsed.hostname or port is None:
            raise ValueError("publish_endpoint must be a TCP URL such as tcp://127.0.0.1:5557")
        for name in ("rate_hz", "retry_delay_s", "status_interval_s"):
            value = getattr(self, name)
            valid = not isinstance(value, bool) and math.isfinite(value)
            invalid_bound = value < 0.0 if name == "retry_delay_s" else value <= 0.0
            valid = valid and not invalid_bound
            if not valid:
                qualifier = "non-negative" if name == "retry_delay_s" else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not isinstance(self.monitor_enabled, bool):
            raise ValueError("monitor_enabled must be a boolean")
        if not self.monitor_host:
            raise ValueError("monitor_host must not be empty")
        if isinstance(self.monitor_port, bool) or not 1 <= self.monitor_port <= 65_535:
            raise ValueError("monitor_port must be between 1 and 65535")


@dataclass(kw_only=True)
class IsaacControllerNodeConfig:
    """Configuration for running Isaac Teleop as a Hub-managed Controller."""

    node_id_path: Path
    teleop: TeleopNodeConfig = field(default_factory=TeleopNodeConfig)
    display_name: str = "Isaac Quest 3 Teleop"
    action_endpoint: str = "tcp://0.0.0.0:5557"
    hub_seed: str | None = None
    presentation: NodePresentation = field(default_factory=NodePresentation)

    def __post_init__(self) -> None:
        self.node_id_path = Path(self.node_id_path)
        if not isinstance(self.teleop, TeleopNodeConfig):
            raise TypeError("teleop must be a TeleopNodeConfig")
        if not isinstance(self.presentation, NodePresentation):
            raise TypeError("presentation must be a NodePresentation")


def make_isaac_controller_node(config: IsaacControllerNodeConfig, runtime: Runtime) -> ControllerNode:
    """Build the generic Controller lifecycle for one Isaac Teleop process."""

    if not isinstance(config, IsaacControllerNodeConfig):
        raise TypeError("config must be an IsaacControllerNodeConfig")
    return ControllerNode(
        ControllerNodeConfig(
            node_id_path=config.node_id_path,
            display_name=config.display_name,
            action_endpoint=config.action_endpoint,
            hub_seed=config.hub_seed,
            action_schemas=(f"{ACTION_SCHEMA}.v{ACTION_SCHEMA_VERSION}",),
            control_modes=("teleop",),
            defer_take_over_until_first_action=True,
            presentation=config.presentation,
        ),
        runtime=runtime,
    )


class MonitorServer:
    """Own a Uvicorn server running alongside the synchronous XR loop."""

    def __init__(self, state: TeleopNodeState, host: str, port: int) -> None:
        config = uvicorn.Config(
            create_monitor_app(state),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="teleop-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 3.0
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._server.started:
            self.stop()
            raise RuntimeError("teleop monitor server did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            raise RuntimeError("teleop monitor server did not stop")


class TeleopNode:
    """Continuously sample one XR session owner and publish atomic actions."""

    def __init__(
        self,
        config: TeleopNodeConfig,
        *,
        control_node: ControllerNode | None = None,
        controller_factory: Callable[[IsaacTeleopConfig], Any] = IsaacXRController,
        publisher_factory: Callable[[str], Any] = ZmqTeleopPublisher,
        monitor_factory: Callable[[TeleopNodeState, str, int], Any] = MonitorServer,
        monotonic: Callable[[], float] = time.monotonic,
        utc_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._control_node = control_node
        self._controller_factory = controller_factory
        self._publisher_factory = publisher_factory
        self._monitor_factory = monitor_factory
        self._monotonic = monotonic
        self._utc_ns = utc_ns
        self._sleep = sleep
        monitor_url = f"http://{config.monitor_host}:{config.monitor_port}"
        self.state = TeleopNodeState(
            session_id=str(uuid.uuid4()),
            publish_endpoint=config.publish_endpoint,
            monitor_url=monitor_url,
            monotonic=monotonic,
            utc_ns=utc_ns,
        )
        self._publisher: Any | None = None
        self._monitor: Any | None = None
        self._engage_authority = EngageAuthority()

    def run(self, *, max_frames: int | None = None, stop_event: threading.Event | None = None) -> None:
        """Run until stopped, or until ``max_frames`` have been sampled for tests."""

        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0
        ):
            raise ValueError("max_frames must be None or a non-negative integer")
        stop_event = threading.Event() if stop_event is None else stop_event
        sampled_total = 0
        try:
            if self._control_node is not None:
                self._control_node.start()
            if self.config.monitor_enabled:
                self._monitor = self._monitor_factory(
                    self.state,
                    self.config.monitor_host,
                    self.config.monitor_port,
                )
                self._monitor.start()
            if self._control_node is None:
                self._publisher = self._publisher_factory(self.config.publish_endpoint)
            self._publish_state("starting")
            if max_frames == 0:
                return

            while not stop_event.is_set():
                session_id = str(uuid.uuid4())
                self.state.begin_session(session_id)
                sequence = 0
                rate_samples: deque[float] = deque(maxlen=max(2, int(self.config.rate_hz * 2)))
                controller: Any | None = None
                try:
                    controller = self._controller_factory(self.config.controller)

                    def report_waiting() -> None:
                        if stop_event.is_set():
                            raise _StopRequestedError
                        self._publish_state("waiting_for_headset")

                    controller.set_connect_wait_callback(report_waiting)
                    self._publish_state("waiting_for_headset")
                    controller.connect()
                    if stop_event.is_set():
                        raise _StopRequestedError
                    self._publish_state("streaming")
                    message = "Quest 3 connected; controller input is streaming."
                    if self.config.monitor_enabled:
                        message += f" Open monitor: {self.state.monitor_url}"
                    print(message, flush=True)
                    next_status_at = self._monotonic()
                    interval_s = 1.0 / self.config.rate_hz
                    while not stop_event.is_set():
                        started_at = self._monotonic()
                        action = controller.get_action()
                        frame = TeleopFrame(
                            session_id=session_id,
                            sequence=sequence,
                            captured_monotonic_ns=int(started_at * 1_000_000_000),
                            captured_utc_ns=self._utc_ns(),
                            action=action,
                        )
                        if self._control_node is not None:
                            self._engage_authority.update(action, self._control_node)
                        published = self._publish_frame(frame)
                        if published:
                            rate_samples.append(started_at)
                        rate_hz = _sample_rate(rate_samples)
                        self.state.record_frame(
                            frame,
                            publish_rate_hz=rate_hz,
                            published=published,
                        )
                        sequence += 1
                        sampled_total += 1
                        if self._publisher is not None and started_at >= next_status_at:
                            self._publisher.publish_status(self.state.snapshot_dict())
                            next_status_at = started_at + self.config.status_interval_s
                        if max_frames is not None and sampled_total >= max_frames:
                            return
                        remaining_s = interval_s - (self._monotonic() - started_at)
                        if remaining_s > 0.0:
                            self._sleep_until_stopped(stop_event, remaining_s)
                except _StopRequestedError:
                    stop_event.set()
                except KeyboardInterrupt:
                    stop_event.set()
                except Exception as error:
                    if not stop_event.is_set():
                        self._publish_state("reconnecting", error=str(error))
                        if self.config.retry_delay_s > 0.0:
                            self._sleep_until_stopped(stop_event, self.config.retry_delay_s)
                finally:
                    if self._control_node is not None:
                        self._engage_authority.reset(self._control_node, release=True)
                    if controller is not None:
                        with contextlib.suppress(Exception):
                            controller.disconnect()
        finally:
            self._publish_state("stopping")
            publisher, self._publisher = self._publisher, None
            if publisher is not None:
                publisher.close()
            monitor, self._monitor = self._monitor, None
            try:
                if monitor is not None:
                    monitor.stop()
            finally:
                self.state.set_state("stopped")
                if self._control_node is not None:
                    self._control_node.stop()

    def _publish_frame(self, frame: TeleopFrame) -> bool:
        """Publish one frame through the selected standalone or managed transport."""

        if self._control_node is not None:
            return self._control_node.publish(
                encode_action_frame(frame),
                captured_monotonic_ns=frame.captured_monotonic_ns,
                captured_utc_ns=frame.captured_utc_ns,
            )
        assert self._publisher is not None
        return self._publisher.publish_action(frame)

    def _publish_state(self, state: str, *, error: str | None = None) -> None:
        self.state.set_state(state, error=error)
        if self._publisher is not None:
            self._publisher.publish_status(self.state.snapshot_dict())

    def _sleep_until_stopped(self, stop_event: threading.Event, duration_s: float) -> None:
        """Sleep in short intervals so process signals remain responsive."""

        remaining_s = duration_s
        while remaining_s > 0.0 and not stop_event.is_set():
            interval_s = min(remaining_s, 0.1)
            self._sleep(interval_s)
            remaining_s -= interval_s


def _sample_rate(samples: deque[float]) -> float:
    if len(samples) < 2:
        return 0.0
    elapsed = samples[-1] - samples[0]
    return 0.0 if elapsed <= 0.0 else (len(samples) - 1) / elapsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--retry-delay-s", type=float, default=2.0)
    parser.add_argument("--monitor-host", default="127.0.0.1")
    parser.add_argument("--monitor-port", type=int, default=8000)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--cloudxr-env-file", default=None)
    parser.add_argument("--cloudxr-install-dir", default=".cloudxr")
    parser.add_argument("--connect-timeout-s", type=float, default=None)
    parser.add_argument("--no-auto-launch", action="store_true")
    parser.add_argument("--no-head-yaw", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = TeleopNodeConfig(
        controller=IsaacTeleopConfig(
            auto_launch_cloudxr=not args.no_auto_launch,
            cloudxr_env_file=args.cloudxr_env_file,
            cloudxr_install_dir=args.cloudxr_install_dir,
            connect_timeout_s=args.connect_timeout_s,
            use_head_yaw=not args.no_head_yaw,
        ),
        publish_endpoint=args.publish_endpoint,
        rate_hz=args.rate_hz,
        retry_delay_s=args.retry_delay_s,
        monitor_enabled=not args.no_monitor,
        monitor_host=args.monitor_host,
        monitor_port=args.monitor_port,
    )
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print("Isaac teleop-node started. No robot is connected or controlled.")
    print(f"Quest client: {_CLIENT_URL}")
    print(f"Action publisher: {config.publish_endpoint}")
    if config.monitor_enabled:
        print(f"Service monitor: http://{config.monitor_host}:{config.monitor_port}")
    TeleopNode(config).run(stop_event=stop_event)


if __name__ == "__main__":
    main()


__all__ = [
    "IsaacControllerNodeConfig",
    "MonitorServer",
    "TeleopNode",
    "TeleopNodeConfig",
    "main",
    "make_isaac_controller_node",
]
