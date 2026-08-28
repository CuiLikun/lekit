"""Three-process command line entry point for the Lekit Control Hub."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import urlsplit

_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "*"})
_PIPER_ROBOT_FIELDS = (
    "id",
    "calibration_dir",
    "channel",
    "interface",
    "bitrate",
    "robot_model",
    "firmware_version",
    "include_gripper",
    "gripper_force_n",
    "gripper_min_width_m",
    "gripper_max_width_m",
    "tcp_offset",
    "eef_workspace_min_m",
    "eef_workspace_max_m",
    "max_eef_target_lead_m",
    "max_eef_target_lead_rad",
    "cartesian_servo",
    "auto_enable",
    "feedback_timeout_s",
    "feedback_poll_interval_s",
    "tcp_feedback_max_age_s",
    "enable_timeout_s",
    "enable_poll_interval_s",
    "speed_percent",
    "disable_on_disconnect",
    "max_relative_target",
    "joint_limit_tolerance_rad",
)
_PIPER_PROCESSOR_FIELDS = (
    "hand",
    "include_gripper",
    "translation_scale",
    "rotation_scale",
    "operator_to_base_rotation",
    "max_translation_from_anchor_m",
    "max_rotation_from_anchor_rad",
    "gripper_min_width_m",
    "gripper_max_width_m",
    "neutral_translation_tolerance_m",
    "neutral_rotation_tolerance_rad",
)


def derive_advertise_endpoint(management_endpoint: str, advertise_host: str | None) -> str:
    """Return a reachable endpoint while preserving the management port."""

    parsed = urlsplit(management_endpoint)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"invalid management endpoint: {management_endpoint!r}") from error
    if parsed.scheme != "tcp" or parsed.hostname is None or port is None:
        raise ValueError("--management-endpoint must be a TCP URL with a host and port")
    host = parsed.hostname
    if host in _WILDCARD_HOSTS and not advertise_host:
        raise ValueError("--advertise-host is required for a wildcard management endpoint")
    if advertise_host:
        candidate = advertise_host.strip().strip("[]")
        if not candidate or "://" in candidate or "/" in candidate:
            raise ValueError("--advertise-host must be a host name or IP address without a port")
        host = candidate
    rendered_host = f"[{host}]" if ":" in host else host
    return f"tcp://{rendered_host}:{port}"


def _advertised_http_url(
    bind_host: str,
    port: int,
    advertise_host: str | None,
    *,
    label: str,
) -> str:
    host = bind_host.strip().strip("[]")
    if host in _WILDCARD_HOSTS and not advertise_host:
        raise ValueError(f"--advertise-host is required for a wildcard {label} host")
    if advertise_host:
        host = advertise_host.strip().strip("[]")
    if not host or "://" in host or "/" in host:
        raise ValueError("--advertise-host must be a host name or IP address without a port")
    rendered_host = f"[{host}]" if ":" in host else host
    return f"http://{rendered_host}:{port}"


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _add_dotted_arguments(parser: argparse.ArgumentParser, prefix: str, fields: Sequence[str]) -> None:
    for field_name in fields:
        parser.add_argument(
            f"--{prefix}.{field_name}",
            dest=f"{prefix}__{field_name}",
            type=_json_value,
            default=argparse.SUPPRESS,
            metavar="VALUE",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the exact three-process CLI without importing hardware adapters."""

    parser = argparse.ArgumentParser(prog="lekit", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    hub = commands.add_parser("hub", help="run scheduling, observability, and the Web console")
    hub.add_argument("--management-endpoint", default="tcp://0.0.0.0:5560")
    hub.add_argument("--advertise-host")
    hub.add_argument("--database", type=Path, default=Path(".lekit/control-hub.sqlite3"))
    hub.add_argument("--web-host", default="127.0.0.1")
    hub.add_argument("--web-port", type=int, default=8080)
    hub.add_argument("--auto-route-single-pair", action="store_true")
    hub.add_argument("--multicast-group", default="239.255.42.99")
    hub.add_argument("--discovery-port", type=int, default=45990)
    hub.set_defaults(runner=_run_hub)

    teleop = commands.add_parser("teleop", help="run the Hub-managed Isaac Quest Controller")
    teleop.add_argument("--node-id-file", type=Path, default=Path(".lekit/nodes/quest3-main"))
    teleop.add_argument("--display-name", default="quest3-main")
    teleop.add_argument(
        "--action-endpoint",
        "--publish-endpoint",
        dest="action_endpoint",
        default="tcp://0.0.0.0:5557",
    )
    teleop.add_argument("--hub-seed")
    teleop.add_argument("--rate-hz", type=float, default=60.0)
    teleop.add_argument("--retry-delay-s", type=float, default=2.0)
    teleop.add_argument("--monitor-host", default="127.0.0.1")
    teleop.add_argument("--monitor-port", type=int, default=8000)
    teleop.add_argument("--advertise-host")
    teleop.add_argument("--no-monitor", action="store_true")
    teleop.add_argument("--cloudxr-env-file")
    teleop.add_argument("--cloudxr-install-dir", default=".cloudxr")
    teleop.add_argument("--connect-timeout-s", type=float)
    teleop.add_argument("--no-auto-launch", action="store_true")
    teleop.add_argument("--no-head-yaw", action="store_true")
    teleop.set_defaults(runner=_run_teleop)

    robot = commands.add_parser("robot", help="run a standard LeRobot Robot behind a RobotNode")
    robot.add_argument("--kind", choices=("piper",), required=True)
    robot.add_argument("--node-id-file", type=Path, default=Path(".lekit/nodes/piper-01"))
    robot.add_argument("--display-name", default="piper-01")
    robot.add_argument("--hub-seed")
    robot.add_argument("--control-rate-hz", type=float, default=60.0)
    robot.add_argument("--enable-motion", action="store_true")
    robot.add_argument("--robot-config", type=Path, help="JSON object for PiperRobotConfig")
    robot.add_argument("--processor-config", type=Path, help="JSON object for Piper processor config")
    robot.add_argument("--video-host", default="127.0.0.1")
    robot.add_argument("--video-port", type=int, default=8081)
    robot.add_argument("--advertise-host")
    _add_dotted_arguments(robot, "robot", _PIPER_ROBOT_FIELDS)
    _add_dotted_arguments(robot, "processor", _PIPER_PROCESSOR_FIELDS)
    robot.set_defaults(runner=_run_robot)
    return parser


@contextlib.contextmanager
def _signal_stop_event() -> Iterator[threading.Event]:
    stop_event = threading.Event()
    previous: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    try:
        yield stop_event
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class _WebServer:
    def __init__(self, app: Any, host: str, port: int) -> None:
        import uvicorn

        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
        )
        self._thread = threading.Thread(target=self._server.run, name="lekit-hub-web", daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 3.0
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._server.started:
            self.stop()
            raise RuntimeError("Hub Web server did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            raise RuntimeError("Hub Web server did not stop within 3 seconds")


def _run_hub(args: argparse.Namespace) -> int:
    from .hub import Hub, HubConfig
    from .web import create_hub_app
    from .zmq_runtime import ZmqRuntime

    advertise_endpoint = derive_advertise_endpoint(args.management_endpoint, args.advertise_host)
    runtime = ZmqRuntime(
        discovery_group=args.multicast_group,
        discovery_port=args.discovery_port,
    )
    hub = Hub(
        HubConfig(
            management_endpoint=args.management_endpoint,
            advertise_endpoint=advertise_endpoint,
            database_path=args.database,
            auto_route_single_pair=args.auto_route_single_pair,
        ),
        runtime=runtime,
    )
    web = _WebServer(create_hub_app(hub), args.web_host, args.web_port)
    try:
        hub.start()
        web.start()
        print(f"Hub management: {advertise_endpoint}", flush=True)
        print(f"Hub Web: http://{args.web_host}:{args.web_port}", flush=True)
        with _signal_stop_event() as stop_event:
            hub.run(stop_event=stop_event)
    finally:
        hub.stop()
        with contextlib.suppress(Exception):
            web.stop()
        runtime.close()
    return 0


def _run_teleop(args: argparse.Namespace) -> int:
    from lekit.teleoperators.isaac_teleop import (
        IsaacControllerNodeConfig,
        IsaacTeleopConfig,
        TeleopNode,
        TeleopNodeConfig,
        make_isaac_controller_node,
    )

    from .model import NodePresentation
    from .zmq_runtime import ZmqRuntime

    runtime = ZmqRuntime()
    teleop_config = TeleopNodeConfig(
        controller=IsaacTeleopConfig(
            auto_launch_cloudxr=not args.no_auto_launch,
            cloudxr_env_file=args.cloudxr_env_file,
            cloudxr_install_dir=args.cloudxr_install_dir,
            connect_timeout_s=args.connect_timeout_s,
            use_head_yaw=not args.no_head_yaw,
        ),
        publish_endpoint=args.action_endpoint,
        rate_hz=args.rate_hz,
        retry_delay_s=args.retry_delay_s,
        monitor_enabled=not args.no_monitor,
        monitor_host=args.monitor_host,
        monitor_port=args.monitor_port,
    )
    controller = make_isaac_controller_node(
        IsaacControllerNodeConfig(
            node_id_path=args.node_id_file,
            teleop=teleop_config,
            display_name=args.display_name,
            action_endpoint=args.action_endpoint,
            hub_seed=args.hub_seed,
            presentation=NodePresentation(
                monitor_url=(
                    _advertised_http_url(
                        args.monitor_host,
                        args.monitor_port,
                        args.advertise_host,
                        label="monitor",
                    )
                    if not args.no_monitor
                    else None
                )
            ),
        ),
        runtime,
    )
    node = TeleopNode(teleop_config, control_node=controller)
    print(f"Controller action endpoint: {args.action_endpoint}", flush=True)
    print(f"Hub seed: {args.hub_seed or 'automatic discovery'}", flush=True)
    try:
        with _signal_stop_event() as stop_event:
            node.run(stop_event=stop_event)
    finally:
        controller.stop()
        runtime.close()
    return 0


def _load_json_object(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _config_values(args: argparse.Namespace, prefix: str, path: Path | None) -> dict[str, Any]:
    values = _load_json_object(path, f"--{prefix}-config")
    marker = f"{prefix}__"
    values.update(
        {name.removeprefix(marker): value for name, value in vars(args).items() if name.startswith(marker)}
    )
    return values


def _build_piper_node(args: argparse.Namespace, runtime: Any) -> Any:
    from lekit.robots.piper import (
        PiperNodeConfig,
        PiperRobotConfig,
        PiperTeleopProcessorConfig,
        _decode_piper_cameras,
        make_piper_robot_node,
        make_piper_video_server,
        piper_video_presentation,
    )
    from lekit.teleoperators.isaac_teleop.protocol import ACTION_SCHEMA, ACTION_SCHEMA_VERSION

    from .robot import RobotNodeConfig

    robot_values = _config_values(args, "robot", args.robot_config)
    processor_values = _config_values(args, "processor", args.processor_config)
    if "calibration_dir" in robot_values and robot_values["calibration_dir"] is not None:
        robot_values["calibration_dir"] = Path(robot_values["calibration_dir"])
    robot_values["cameras"] = _decode_piper_cameras(robot_values.pop("cameras", {}))
    robot_config = PiperRobotConfig(**robot_values)
    video = None
    node_config = RobotNodeConfig(
        node_id_path=args.node_id_file,
        display_name=args.display_name,
        accepted_payload_schemas=(f"{ACTION_SCHEMA}.v{ACTION_SCHEMA_VERSION}",),
        control_enabled=args.enable_motion,
        control_rate_hz=args.control_rate_hz,
        hub_seed=args.hub_seed,
    )
    if robot_config.cameras:
        video = make_piper_video_server(
            robot_config,
            host=args.video_host,
            port=args.video_port,
            advertise_host=args.advertise_host,
        )
        node_config = replace(node_config, presentation=piper_video_presentation(video))
    node = make_piper_robot_node(
        PiperNodeConfig(
            node=node_config,
            robot=robot_config,
            processor=PiperTeleopProcessorConfig(**processor_values),
            enable_motion=args.enable_motion,
            observation_sinks=(video,) if video is not None else (),
        ),
        runtime,
    )
    return node, video


def _run_robot(args: argparse.Namespace) -> int:
    from .zmq_runtime import ZmqRuntime

    runtime = ZmqRuntime()
    node, video = _build_piper_node(args, runtime)
    mode = "MOTION ENABLED" if args.enable_motion else "READ-ONLY (motion disabled)"
    print(f"Robot {args.display_name}: {mode}", flush=True)
    print(f"Hub seed: {args.hub_seed or 'automatic discovery'}", flush=True)
    if args.enable_motion:
        print("Motion still requires Hub assignment and an explicit take-over.", flush=True)
    period_s = 1.0 / args.control_rate_hz
    try:
        with _signal_stop_event() as stop_event:
            if video is not None:
                video.start()
                print(f"Robot video: http://{args.advertise_host or args.video_host}:{args.video_port}", flush=True)
            node.start()
            while not stop_event.is_set():
                started = time.monotonic()
                node.run_cycle()
                stop_event.wait(max(0.0, period_s - (time.monotonic() - started)))
    finally:
        # RobotNode.stop enters local HOLD before disconnecting the LeRobot Robot.
        try:
            node.stop()
        finally:
            try:
                if video is not None:
                    video.stop()
            finally:
                runtime.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and run one of the three independently deployable processes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    runner: Callable[[argparse.Namespace], int] = args.runner
    try:
        return runner(args)
    except ValueError as error:
        parser.error(str(error))
    return 2


__all__ = ["build_parser", "derive_advertise_endpoint", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
