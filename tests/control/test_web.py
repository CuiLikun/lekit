"""Representative boundary tests for the Hub's browser-facing service."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from lekit.control.hub import ControlConflict, IncompatibleNode, NodeUnavailable
from lekit.control.web import create_hub_app


class FakeHub:
    """Small synchronous stand-in exposing only the public Hub surface."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._condition = threading.Condition()
        self._snapshot = {"version": 1, "hub_epoch": "epoch-1", "nodes": [], "controls": [], "alerts": []}

    def assign(self, robot: str, controller: str, *, control_mode: str, actor: str | None) -> dict[str, str]:
        self.calls.append(("assign", robot, controller, control_mode, actor))
        return {"handle_id": "handle-1"}

    def request_take_over(self, handle: str, *, actor: str | None) -> None:
        self.calls.append(("take-over", handle, actor))

    def request_hand_over(self, handle: str, *, actor: str | None) -> None:
        self.calls.append(("hand-over", handle, actor))

    def renew(self, handle: str, *, actor: str | None) -> dict[str, str]:
        self.calls.append(("renew", handle, actor))
        return {"handle_id": handle}

    def revoke(self, handle: str, *, reason: str, actor: str | None) -> None:
        self.calls.append(("revoke", handle, reason, actor))

    def force_hold(self, robot: str, *, reason: str, actor: str | None) -> None:
        self.calls.append(("force-hold", robot, reason, actor))

    def get_snapshot(self) -> dict[str, object]:
        return self._snapshot

    def list_nodes(self) -> list[object]:
        return []

    def list_history(self, *, limit: int) -> list[object]:
        self.calls.append(("history", limit))
        return []

    def watch(self, after_version: int, timeout_s: float) -> dict[str, object]:
        with self._condition:
            self._condition.wait_for(lambda: self._snapshot["version"] > after_version, timeout_s)
            return self._snapshot

    def publish(self, *, version: int) -> None:
        with self._condition:
            self._snapshot = {**self._snapshot, "version": version}
            self._condition.notify_all()


@pytest.fixture
def fake_hub() -> FakeHub:
    return FakeHub()


@pytest.fixture
def client(fake_hub: FakeHub) -> TestClient:
    return TestClient(create_hub_app(fake_hub))


def test_assignment_forwards_operator_to_public_hub_interface(client: TestClient, fake_hub: FakeHub) -> None:
    """Would fail if the browser endpoint did not make an attributable assignment."""
    response = client.post(
        "/api/assign",
        json={"robot": "piper-01", "controller": "quest3-main", "control_mode": "teleop"},
        headers={"X-Operator": "operator-1"},
    )

    assert response.status_code == 201
    assert response.json() == {"handle_id": "handle-1"}
    assert fake_hub.calls[-1] == ("assign", "piper-01", "quest3-main", "teleop", "operator-1")
    oversized = client.post(
        "/api/assign",
        json={"robot": "piper-01", "controller": "quest3-main"},
        headers={"X-Operator": "x" * 129},
    )
    assert oversized.status_code == 422


def test_control_commands_use_client_address_and_validate_safety_reason(
    client: TestClient, fake_hub: FakeHub
) -> None:
    """Would fail if unaudited operator actions or empty hold reasons were accepted."""
    invalid = client.post("/api/robots/piper-01/force-hold", json={"reason": ""})
    valid = client.post("/api/robots/piper-01/force-hold", json={"reason": "operator check"})

    assert invalid.status_code == 422
    assert valid.status_code == 204
    assert fake_hub.calls[-1] == ("force-hold", "piper-01", "operator check", "testclient")


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (KeyError("missing"), 404),
        (NodeUnavailable("unknown Robot piper-01"), 404),
        (NodeUnavailable("Robot piper-01 is in SAFETY"), 422),
        (ControlConflict("busy"), 409),
        (IncompatibleNode("unsafe"), 422),
    ],
)
def test_domain_errors_are_exposed_as_actionable_http_statuses(
    client: TestClient, fake_hub: FakeHub, error: Exception, expected_status: int
) -> None:
    """Would fail if an operator saw a server error for a known Hub refusal."""

    def rejected_assign(*args: object, **kwargs: object) -> object:
        raise error

    fake_hub.assign = rejected_assign  # type: ignore[method-assign]

    response = client.post("/api/assign", json={"robot": "piper-01", "controller": "quest3-main"})

    assert response.status_code == expected_status
    assert response.json()["detail"] == str(error)


def test_websocket_emits_initial_and_new_snapshot_versions(client: TestClient, fake_hub: FakeHub) -> None:
    """Would fail if the live view replayed an already delivered snapshot."""
    with client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["version"] == 1
        fake_hub.publish(version=2)
        assert socket.receive_json()["version"] == 2


def test_root_serves_fixed_layout_ui_with_required_control_routes(client: TestClient) -> None:
    """Would fail if a packaged UI lost operational columns or a control route."""
    response = client.get("/")

    assert response.status_code == 200
    for label in (
        "Node",
        "Role",
        "Capabilities",
        "Runtime",
        "Control",
        "Session",
        "Assignment",
        "Handle",
        "TTL",
        "Rate",
        "Frame age",
        "Sequence",
        "Connected",
        "Tracking",
        "Engaged",
        "Processor",
        "Error",
        "Actions",
    ):
        assert label in response.text
    assert "table-layout: fixed" in response.text
    assert "onclick=" not in response.text
    assert "data-action" in response.text
    assert "robot_safety_with_active_desire" in response.text
    assert "/api/handles/" in response.text
    assert "/api/robots/" in response.text
