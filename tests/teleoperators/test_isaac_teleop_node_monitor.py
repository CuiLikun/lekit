from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lekit.teleoperators.isaac_teleop.node_state import TeleopNodeState, create_monitor_app
from lekit.teleoperators.isaac_teleop.protocol import TeleopFrame, neutral_action
from lekit.teleoperators.isaac_teleop.teleop_node import MonitorServer

_NODE = shutil.which("node")
_TRAJECTORY_MODULE = (
    Path(__file__).parents[2] / "src/lekit/teleoperators/isaac_teleop/teleop_node_monitor_trajectory.js"
)


class ManualTime:
    def __init__(self) -> None:
        self.monotonic_s = 10.0
        self.utc_ns = 1_000_000_000

    def monotonic(self) -> float:
        return self.monotonic_s

    def time_ns(self) -> int:
        return self.utc_ns


def state() -> tuple[TeleopNodeState, ManualTime]:
    clock = ManualTime()
    store = TeleopNodeState(
        session_id="session-monitor",
        publish_endpoint="tcp://127.0.0.1:5557",
        monitor_url="http://127.0.0.1:8000",
        monotonic=clock.monotonic,
        utc_ns=clock.time_ns,
    )
    return store, clock


def run_trajectory_script(source: str) -> dict:
    result = subprocess.run(
        [_NODE, "--input-type=module", "--eval", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_status_route_exposes_service_metrics_and_every_action_field() -> None:
    store, clock = state()
    action = neutral_action()
    action["left.trigger"] = 0.25
    action["right.is_tracking"] = True
    store.set_state("streaming")
    store.record_frame(
        TeleopFrame("session-monitor", 12, 20, clock.utc_ns, action),
        publish_rate_hz=59.5,
    )
    clock.monotonic_s += 0.04

    response = TestClient(create_monitor_app(store)).get("/api/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "streaming"
    assert payload["session_id"] == "session-monitor"
    assert payload["sequence"] == 12
    assert payload["sampled_frames"] == 1
    assert payload["published_frames"] == 1
    assert payload["dropped_frames"] == 0
    assert payload["publish_rate_hz"] == 59.5
    assert payload["last_frame_age_ms"] == 40.0
    assert payload["uptime"] == "00:00:00"
    assert payload["publish_endpoint"] == "tcp://127.0.0.1:5557"
    assert payload["action"]["left.trigger"] == 0.25
    assert payload["action"]["right.is_tracking"] is True
    assert len(payload["action"]) == 28


def test_websocket_pushes_the_latest_controller_state() -> None:
    store, clock = state()
    client = TestClient(create_monitor_app(store))

    with client.websocket_connect("/api/stream") as websocket:
        initial = websocket.receive_json()
        assert initial["state"] == "starting"

        for sequence in range(10):
            action = neutral_action()
            action["right.trigger"] = sequence / 10
            store.record_frame(
                TeleopFrame("session-monitor", sequence, 20 + sequence, clock.utc_ns, action),
                publish_rate_hz=60.0,
            )

        latest = websocket.receive_json()

    assert latest["sequence"] == 9
    assert latest["action"]["right.trigger"] == pytest.approx(0.9)


def test_websocket_status_updates_are_capped_at_sixty_hertz() -> None:
    store, _clock = state()
    client = TestClient(create_monitor_app(store))

    with client.websocket_connect("/api/stream") as websocket:
        websocket.receive_json()
        started_at = time.monotonic()
        for sequence in range(4):
            store.begin_session(f"session-{sequence}")
            websocket.receive_json()
        elapsed_s = time.monotonic() - started_at

    assert elapsed_s >= 0.045


@pytest.mark.skipif(_NODE is None, reason="Node.js is required to execute the browser trajectory module")
def test_engagement_trajectory_is_retained_until_the_next_engage() -> None:
    result = run_trajectory_script(
        f"""
        import {{ EngagementTrajectory }} from {json.dumps(_TRAJECTORY_MODULE.as_uri())};
        const trajectory = new EngagementTrajectory({{ maximumPoints: 3, minimumDistance: 0.004 }});
        trajectory.update(true, [0, 0, 0]);
        trajectory.update(true, [0.003, 0, 0]);
        trajectory.update(true, [0.005, 0, 0]);
        trajectory.update(false, [9, 9, 9]);
        const retained = trajectory.points;
        trajectory.update(true, [1, 0, 0]);
        console.log(JSON.stringify({{ retained, restarted: trajectory.points }}));
        """
    )

    assert result == {
        "retained": [[0, 0, 0], [0.005, 0, 0]],
        "restarted": [[1, 0, 0]],
    }


@pytest.mark.skipif(_NODE is None, reason="Node.js is required to execute the browser trajectory module")
def test_engagement_trajectory_discards_its_oldest_points_at_capacity() -> None:
    result = run_trajectory_script(
        f"""
        import {{ EngagementTrajectory }} from {json.dumps(_TRAJECTORY_MODULE.as_uri())};
        const trajectory = new EngagementTrajectory({{ maximumPoints: 3, minimumDistance: 0 }});
        for (let x = 0; x < 4; x += 1) trajectory.update(true, [x, 0, 0]);
        console.log(JSON.stringify({{ points: trajectory.points }}));
        """
    )

    assert result == {"points": [[1, 0, 0], [2, 0, 0], [3, 0, 0]]}


def test_monitor_root_is_a_read_only_live_dashboard() -> None:
    store, _clock = state()
    client = TestClient(create_monitor_app(store))

    response = client.get("/")

    assert response.status_code == 200
    assert "Isaac Teleop Node" in response.text
    assert "/api/stream" in response.text
    assert 'id="monitor-url"' not in response.text
    assert 'id="session"' not in response.text
    assert 'id="endpoint"' not in response.text
    assert 'aria-label="Left Quest 3 controller"' in response.text
    assert 'aria-label="Right Quest 3 controller"' in response.text
    assert 'data-control="primary_button"' in response.text
    assert 'data-control="secondary_button"' in response.text
    assert 'data-control="menu_button"' in response.text
    assert 'data-control="thumbstick"' in response.text
    assert 'data-control="trigger"' in response.text
    assert 'data-control="squeeze"' in response.text
    assert 'id="pose-scene"' in response.text
    assert 'aria-label="3D relative controller poses"' in response.text
    assert 'id="render-rate"' in response.text
    assert "https://" not in response.text
    assert client.post("/api/status").status_code == 405


def test_monitor_header_keeps_title_status_and_metrics_in_one_row() -> None:
    store, _clock = state()
    response = TestClient(create_monitor_app(store)).get("/")

    header_end = response.text.index("</header>")
    assert response.text.index("<h1>Isaac Teleop Node</h1>") < header_end
    assert response.text.index('id="badge"') < header_end
    assert response.text.index('<section class="metrics"') < header_end
    assert '<div class="subtitle">' not in response.text


def test_monitor_serves_all_visualization_assets_locally() -> None:
    store, _clock = state()
    client = TestClient(create_monitor_app(store))

    script = client.get("/assets/teleop_node_monitor.js")
    trajectory = client.get("/assets/teleop_node_monitor_trajectory.js")
    three = client.get("/assets/three.module.min.js")
    three_core = client.get("/assets/three.core.min.js")
    license_file = client.get("/assets/three.LICENSE.txt")

    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert trajectory.status_code == 200
    assert "javascript" in trajectory.headers["content-type"]
    assert three.status_code == 200
    assert "javascript" in three.headers["content-type"]
    assert len(three.content) > 300_000
    assert b'from"./three.core.min.js"' in three.content
    assert three_core.status_code == 200
    assert "javascript" in three_core.headers["content-type"]
    assert len(three_core.content) > 300_000
    assert license_file.status_code == 200
    assert "MIT License" in license_file.text


def test_uptime_uses_readable_duration_including_days() -> None:
    store, clock = state()
    clock.monotonic_s += 90_061.9

    assert store.snapshot_dict()["uptime"] == "1d 01:01:01"


def test_health_route_reports_process_liveness_and_fault_detail() -> None:
    store, _clock = state()
    client = TestClient(create_monitor_app(store))
    assert client.get("/healthz").json() == {"ok": True, "state": "starting"}

    store.set_state("reconnecting", error="OpenXR IPC disconnected")

    assert client.get("/healthz").json() == {"ok": True, "state": "reconnecting"}
    assert client.get("/api/status").json()["last_error"] == "OpenXR IPC disconnected"


def test_waiting_state_has_safe_neutral_values_before_the_first_frame() -> None:
    store, _clock = state()
    store.set_state("waiting_for_headset")

    payload = store.snapshot_dict()

    assert payload["sequence"] is None
    assert payload["last_frame_age_ms"] is None
    assert payload["published_frames"] == 0
    assert payload["sampled_frames"] == 0
    assert payload["dropped_frames"] == 0
    assert payload["action"]["left.is_tracking"] is False
    assert payload["action"]["right.is_engaged"] is False


def test_monitor_stop_reports_a_thread_that_does_not_exit() -> None:
    class StuckThread:
        def is_alive(self) -> bool:
            return True

        def join(self, *, timeout: float) -> None:
            assert timeout == 3.0

    monitor = object.__new__(MonitorServer)
    monitor._server = SimpleNamespace(should_exit=False)
    monitor._thread = StuckThread()

    with pytest.raises(RuntimeError, match="did not stop"):
        monitor.stop()

    assert monitor._server.should_exit is True
