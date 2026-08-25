"""Connect a Quest 3 controller and inspect its input stream in Rerun.

Run this module directly. It never creates a robot, sends a robot command, or
writes a dataset. Closing it only stops the XR session and the local viewer.
"""

from __future__ import annotations

import argparse
import socket
import time
from collections import deque
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote, urlparse

import numpy as np

from .config import IsaacTeleopConfig
from .xr_controller import CONTROLLER_SIDES, IsaacXRController

_CLIENT_URL = "https://nvidia.github.io/IsaacTeleop/client"
_DEFAULT_RATE_HZ = 60.0
_TRACE_LENGTH = 256


def _neutral_action() -> dict[str, Any]:
    """Return a visible startup frame before XR tracking is available."""

    action: dict[str, Any] = {}
    for side in CONTROLLER_SIDES:
        action.update(
            {
                f"{side}.translation": np.zeros(3, dtype=np.float32),
                f"{side}.rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                f"{side}.aim_translation": np.zeros(3, dtype=np.float32),
                f"{side}.aim_rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                f"{side}.thumbstick": np.zeros(2, dtype=np.float32),
                f"{side}.is_engaged": False,
                f"{side}.is_tracking": False,
                f"{side}.is_aim_tracking": False,
                f"{side}.squeeze": 0.0,
                f"{side}.trigger": 0.0,
                f"{side}.thumbstick_click": 0.0,
                f"{side}.primary_button": 0.0,
                f"{side}.secondary_button": 0.0,
                f"{side}.menu_button": 0.0,
            }
        )
    return action


def _status_lines(action: dict[str, Any], published_samples: int) -> list[str]:
    """Build a complete action snapshot for the Live status view."""

    lines = ["# Quest 3 input test", f"- Published samples: **{published_samples}**"]
    for side in CONTROLLER_SIDES:
        prefix = f"{side}."
        translation = np.asarray(action[f"{prefix}translation"], dtype=np.float32)
        rotation = np.asarray(action[f"{prefix}rotation"], dtype=np.float32)
        aim_translation = np.asarray(action[f"{prefix}aim_translation"], dtype=np.float32)
        aim_rotation = np.asarray(action[f"{prefix}aim_rotation"], dtype=np.float32)
        thumbstick = np.asarray(action[f"{prefix}thumbstick"], dtype=np.float32)
        lines.extend(
            (
                f"## {side.title()} controller",
                f"- `{prefix}translation`: `[{translation[0]:+.3f}, {translation[1]:+.3f}, "
                f"{translation[2]:+.3f}]` m",
                f"- `{prefix}rotation`: `[{rotation[0]:+.3f}, {rotation[1]:+.3f}, "
                f"{rotation[2]:+.3f}, {rotation[3]:+.3f}]` xyzw",
                f"- `{prefix}aim_translation`: `[{aim_translation[0]:+.3f}, "
                f"{aim_translation[1]:+.3f}, {aim_translation[2]:+.3f}]` m",
                f"- `{prefix}aim_rotation`: `[{aim_rotation[0]:+.3f}, {aim_rotation[1]:+.3f}, "
                f"{aim_rotation[2]:+.3f}, {aim_rotation[3]:+.3f}]` xyzw",
                f"- `{prefix}squeeze`: `{float(action[f'{prefix}squeeze']):+.3f}`",
                f"- `{prefix}trigger`: `{float(action[f'{prefix}trigger']):+.3f}`",
                f"- `{prefix}thumbstick`: `[{thumbstick[0]:+.3f}, {thumbstick[1]:+.3f}]`",
                f"- `{prefix}thumbstick_click`: `{float(action[f'{prefix}thumbstick_click']):.0f}`",
                f"- `{prefix}primary_button`: `{float(action[f'{prefix}primary_button']):.0f}`",
                f"- `{prefix}secondary_button`: `{float(action[f'{prefix}secondary_button']):.0f}`",
                f"- `{prefix}menu_button`: `{float(action[f'{prefix}menu_button']):.0f}`",
                f"- `{prefix}is_tracking`: **{bool(action[f'{prefix}is_tracking'])}**",
                f"- `{prefix}is_aim_tracking`: **{bool(action[f'{prefix}is_aim_tracking'])}**",
                f"- `{prefix}is_engaged`: **{bool(action[f'{prefix}is_engaged'])}**",
            )
        )
    return lines


def _primary_ipv4() -> str | None:
    """Find the outbound IPv4 address without sending any network traffic."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
        except OSError:
            return None
        return str(sock.getsockname()[0])


def _verify_rerun_endpoint(rerun_url: str, *, timeout_s: float = 3.0) -> None:
    """Fail clearly when the remote Rerun proxy is not reachable."""

    parsed = urlparse(rerun_url)
    if parsed.scheme not in {"rerun+http", "rerun+https"} or parsed.hostname is None:
        raise ValueError(f"Invalid Rerun proxy URL: {rerun_url!r}")
    default_port = 443 if parsed.scheme == "rerun+https" else 9876
    try:
        port = parsed.port or default_port
    except ValueError as error:
        raise ValueError(f"Invalid Rerun proxy URL: {rerun_url!r}") from error
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_s):
            pass
    except OSError as error:
        raise ConnectionError(
            f"Cannot reach Rerun at {parsed.hostname}:{port}. Start the Viewer/server before this program."
        ) from error


def _blueprint(rr: Any) -> Any:
    return rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial3DView(
                origin="/",
                contents=["/quest3/controllers/**"],
                name="Relative controller poses",
                eye_controls=rr.blueprint.EyeControls3D(
                    position=(0.35, -0.45, 0.30),
                    look_target=(0.0, 0.0, 0.0),
                    eye_up=(0.0, 0.0, 1.0),
                ),
            ),
            rr.blueprint.TextDocumentView(
                origin="/",
                contents=["/quest3/status"],
                name="Live status",
            ),
        )
    )


class Quest3Visualizer:
    """Publish standalone XR action data into a local Rerun Viewer."""

    def __init__(
        self,
        *,
        spawn_viewer: bool,
        rerun_url: str | None = None,
        serve_web: bool = False,
        grpc_port: int = 9876,
        web_port: int = 9090,
    ):
        import rerun as rr

        if rerun_url is not None and serve_web:
            raise ValueError("rerun_url and serve_web are mutually exclusive")
        self._rr = rr
        self._traces = {side: deque(maxlen=_TRACE_LENGTH) for side in CONTROLLER_SIDES}
        self._started_at = time.monotonic()
        self._published_samples = 0
        blueprint = _blueprint(rr)
        if rerun_url is not None:
            _verify_rerun_endpoint(rerun_url)
        rr.init(
            "lekit_quest3_input_test",
            spawn=False,
            default_blueprint=blueprint,
        )
        if rerun_url is not None:
            rr.connect_grpc(rerun_url)
            rr.send_blueprint(blueprint, make_active=True, make_default=True)
            print(f"Rerun {rr.__version__} connected to {rerun_url}; blueprint activated", flush=True)
        elif serve_web:
            server_url = rr.serve_grpc(grpc_port=grpc_port, default_blueprint=blueprint)
            rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=server_url)
            rr.send_blueprint(blueprint, make_active=True, make_default=True)
            viewer_url = f"http://127.0.0.1:{web_port}/?url={quote(server_url, safe='')}"
            print(
                f"Rerun {rr.__version__} Web Viewer ready at {viewer_url}; blueprint activated",
                flush=True,
            )
        elif spawn_viewer:
            rr.spawn(default_blueprint=blueprint)
        rr.log("/quest3/controllers", rr.ViewCoordinates.RFU, static=True)
        rr.log(
            "/quest3/controllers/axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]] * 3,
                vectors=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                colors=[[230, 80, 80], [70, 180, 110], [75, 125, 230]],
                labels=["right +X", "forward +Y", "up +Z"],
            ),
            static=True,
        )
        for side, color in (("left", [70, 180, 230]), ("right", [235, 185, 55])):
            rr.log(
                f"/quest3/controllers/{side}/grip/marker",
                rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=[0.012], labels=[f"{side} grip"]),
                static=True,
            )
            rr.log(
                f"/quest3/controllers/{side}/aim/ray",
                rr.Arrows3D(
                    origins=[[0.0, 0.0, 0.0]],
                    vectors=[[0.0, 0.08, 0.0]],
                    colors=[color],
                    labels=[f"{side} aim"],
                ),
                static=True,
            )
        self.log_action(_neutral_action())
        self.log_status("Rerun connected. Waiting for the XR session...")
        self.log_action(_neutral_action())
        self.flush()

    @property
    def published_samples(self) -> int:
        return self._published_samples

    def flush(self, *, timeout_s: float = 5.0) -> None:
        """Push buffered samples to a remote Viewer."""

        recording = self._rr.get_data_recording()
        if recording is not None:
            recording.flush(timeout_sec=timeout_s)

    def log_status(self, message: str) -> None:
        """Publish a human-readable lifecycle status to the status panel."""

        self._rr.log(
            "/quest3/status",
            self._rr.TextDocument(f"# Quest 3 input test\n\n{message}"),
            static=True,
        )

    def log_action(self, action: dict[str, Any]) -> None:
        """Log one normalized teleoperator action frame."""

        rr = self._rr
        self._published_samples += 1
        rr.set_time("control_time", duration=time.monotonic() - self._started_at)
        status_lines = _status_lines(action, self._published_samples)
        for side in CONTROLLER_SIDES:
            prefix = f"{side}."
            translation = np.asarray(action[f"{prefix}translation"], dtype=np.float32)
            rotation = np.asarray(action[f"{prefix}rotation"], dtype=np.float32)
            aim_translation = np.asarray(action[f"{prefix}aim_translation"], dtype=np.float32)
            aim_rotation = np.asarray(action[f"{prefix}aim_rotation"], dtype=np.float32)
            engaged = bool(action[f"{prefix}is_engaged"])
            rr.log(
                f"/quest3/controllers/{side}/grip",
                rr.Transform3D(translation=translation, quaternion=rotation),
            )
            rr.log(
                f"/quest3/controllers/{side}/aim",
                rr.Transform3D(translation=aim_translation, quaternion=aim_rotation),
            )
            trace = self._traces[side]
            if engaged:
                trace.append(translation.copy())
            else:
                trace.clear()
            if len(trace) >= 2:
                color = [70, 180, 230] if side == "left" else [235, 185, 55]
                rr.log(
                    f"/quest3/controllers/{side}/trace",
                    rr.LineStrips3D([list(trace)], colors=[color], radii=[0.002]),
                )
        rr.log(
            "/quest3/status",
            rr.TextDocument("\n".join(status_lines)),
            static=True,
        )


def run(
    config: IsaacTeleopConfig,
    *,
    rate_hz: float = _DEFAULT_RATE_HZ,
    duration_s: float | None = None,
    spawn_viewer: bool = True,
    rerun_url: str | None = None,
    serve_web: bool = False,
    grpc_port: int = 9876,
    web_port: int = 9090,
) -> None:
    """Read a real controller and show its input without touching a robot."""

    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    viewer = Quest3Visualizer(
        spawn_viewer=spawn_viewer,
        rerun_url=rerun_url,
        serve_web=serve_web,
        grpc_port=grpc_port,
        web_port=web_port,
    )
    controller = IsaacXRController(config)
    controller.set_connect_wait_callback(lambda: viewer.log_action(_neutral_action()))
    deadline = time.monotonic() + duration_s if duration_s is not None else None
    interval_s = 1.0 / rate_hz
    print("Quest 3 input test started. No robot is connected or controlled.")
    viewer.log_status("Connecting to CloudXR/OpenXR. Waiting for Quest 3...")
    try:
        with controller:
            print("XR session connected; streaming controller input.", flush=True)
            viewer.log_status("XR session connected. Reading controller input...")
            received_frames = 0
            last_report_at = time.monotonic()
            while deadline is None or time.monotonic() < deadline:
                started_at = time.monotonic()
                action = controller.get_action()
                viewer.log_action(action)
                received_frames += 1
                if started_at - last_report_at >= 1.0:
                    print(
                        "Quest frames: "
                        f"{received_frames}; "
                        f"left(tracking={bool(action['left.is_tracking'])}, "
                        f"trigger={float(action['left.trigger']):.3f}, "
                        f"squeeze={float(action['left.squeeze']):.3f}); "
                        f"right(tracking={bool(action['right.is_tracking'])}, "
                        f"trigger={float(action['right.trigger']):.3f}, "
                        f"squeeze={float(action['right.squeeze']):.3f})",
                        flush=True,
                    )
                    viewer.flush(timeout_s=1.0)
                    last_report_at = started_at
                remaining_s = interval_s - (time.monotonic() - started_at)
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-hz", type=float, default=_DEFAULT_RATE_HZ)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--cloudxr-env-file", default=None)
    parser.add_argument("--cloudxr-install-dir", default=".cloudxr")
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=None,
        help="Stop waiting for a Quest connection after this many seconds (default: wait indefinitely)",
    )
    parser.add_argument("--no-auto-launch", action="store_true")
    parser.add_argument("--no-head-yaw", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--serve-web",
        action="store_true",
        help="Host a headless Rerun gRPC proxy and Web Viewer on this workstation",
    )
    parser.add_argument("--rerun-grpc-port", type=int, default=9876)
    parser.add_argument("--rerun-web-port", type=int, default=9090)
    parser.add_argument(
        "--rerun-url",
        default=None,
        help="Rerun gRPC proxy URL, for example rerun+http://127.0.0.1:9876/proxy",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the real-device input test and visualizer."""

    args = _parser().parse_args(argv)
    if args.duration_s is not None and args.duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if args.rerun_url is not None and args.serve_web:
        raise ValueError("--rerun-url and --serve-web are mutually exclusive")
    if not 1 <= args.rerun_grpc_port <= 65535 or not 1 <= args.rerun_web_port <= 65535:
        raise ValueError("Rerun ports must be between 1 and 65535")
    ip_address = _primary_ipv4()
    print(f"In Quest 3, open {_CLIENT_URL}")
    if ip_address is not None:
        print(f"Connect it to this workstation at {ip_address}.")
    else:
        print("Connect it to this workstation's LAN address.")
    run(
        IsaacTeleopConfig(
            auto_launch_cloudxr=not args.no_auto_launch,
            cloudxr_env_file=args.cloudxr_env_file,
            cloudxr_install_dir=args.cloudxr_install_dir,
            connect_timeout_s=args.connect_timeout_s,
            use_head_yaw=not args.no_head_yaw,
        ),
        rate_hz=args.rate_hz,
        duration_s=args.duration_s,
        spawn_viewer=not args.no_viewer and args.rerun_url is None and not args.serve_web,
        rerun_url=args.rerun_url,
        serve_web=args.serve_web,
        grpc_port=args.rerun_grpc_port,
        web_port=args.rerun_web_port,
    )


if __name__ == "__main__":
    main()
