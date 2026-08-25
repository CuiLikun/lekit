"""Rerun blueprint contract tests for the standalone Quest visualizer."""

from __future__ import annotations

import pytest
import rerun as rr

from lekit.teleoperators.isaac_teleop.quest3_visualizer import (
    Quest3Visualizer,
    _blueprint,
    _neutral_action,
    _status_lines,
    _verify_rerun_endpoint,
)


def test_blueprint_contains_only_relative_poses_and_live_status_views() -> None:
    blueprint = _blueprint(rr)
    views = blueprint.root_container.contents

    assert len(views) == 2
    spatial, status = views
    assert spatial.name == "Relative controller poses"
    assert spatial.origin == "/"
    assert spatial.contents == ["/quest3/controllers/**"]
    assert status.name == "Live status"
    assert status.origin == "/"
    assert status.contents == ["/quest3/status"]


def test_startup_action_populates_every_status_field() -> None:
    action = _neutral_action()

    for side in ("left", "right"):
        assert action[f"{side}.translation"].shape == (3,)
        assert action[f"{side}.rotation"].tolist() == [0.0, 0.0, 0.0, 1.0]
        assert action[f"{side}.aim_translation"].shape == (3,)
        assert action[f"{side}.aim_rotation"].tolist() == [0.0, 0.0, 0.0, 1.0]
        assert action[f"{side}.thumbstick"].shape == (2,)
        assert not action[f"{side}.is_tracking"]
        assert not action[f"{side}.is_aim_tracking"]
        assert not action[f"{side}.is_engaged"]


def test_live_status_contains_every_action_feature_by_full_key() -> None:
    lines = "\n".join(_status_lines(_neutral_action(), published_samples=1))

    fields = (
        "translation",
        "rotation",
        "aim_translation",
        "aim_rotation",
        "squeeze",
        "trigger",
        "thumbstick",
        "thumbstick_click",
        "primary_button",
        "secondary_button",
        "menu_button",
        "is_tracking",
        "is_aim_tracking",
        "is_engaged",
    )
    for side in ("left", "right"):
        for field in fields:
            assert lines.count(f"`{side}.{field}`:") == 1


def test_rerun_endpoint_preflight_uses_url_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return None

    def create_connection(address, *, timeout):
        requested.append((address, timeout))
        return _Connection()

    monkeypatch.setattr("socket.create_connection", create_connection)

    _verify_rerun_endpoint("rerun+http://192.168.5.31:9876/proxy", timeout_s=1.5)

    assert requested == [(("192.168.5.31", 9876), 1.5)]


def test_rerun_endpoint_preflight_reports_unreachable_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    def create_connection(_address, *, timeout):
        raise TimeoutError(timeout)

    monkeypatch.setattr("socket.create_connection", create_connection)

    with pytest.raises(ConnectionError, match="Start the Viewer/server"):
        _verify_rerun_endpoint("rerun+http://192.168.5.31:9876/proxy")


def test_web_server_and_remote_endpoint_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Quest3Visualizer(
            spawn_viewer=False,
            rerun_url="rerun+http://127.0.0.1:9876/proxy",
            serve_web=True,
        )
