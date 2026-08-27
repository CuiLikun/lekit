from __future__ import annotations

import time
from pathlib import Path

from lekit.control import (
    ControllerNode,
    ControllerNodeConfig,
    HandleState,
    Hub,
    HubConfig,
    MemoryRuntime,
    RobotControlState,
    RobotNode,
    RobotNodeConfig,
)
from lekit.control.controller import HandleNotGranted
from lerobot.robots.robot import Robot


class FakeRobot(Robot):
    name = "fake"
    config_class = object

    def __init__(self) -> None:
        self.connected = False
        self.sent: list[dict[str, float]] = []

    @property
    def observation_features(self) -> dict[str, type]:
        return {"x": float}

    @property
    def action_features(self) -> dict[str, type]:
        return {"x": float}

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.connected = True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, float]:
        return {"x": 0.0}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        sent = dict(action)
        self.sent.append(sent)
        return sent

    def disconnect(self) -> None:
        self.connected = False


class FloatProcessor:
    accepted_payload_schemas = frozenset({"test.float.v1"})

    def __call__(self, payload: bytes, observation: dict[str, float]) -> dict[str, float]:
        del observation
        return {"x": float(payload.decode("ascii"))}

    def reset(self) -> None:
        return None


def _write_id(path: Path, value: str) -> Path:
    path.write_text(f"{value}\n", encoding="utf-8")
    return path


def _pump(hub: Hub, predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hub.run_once(timeout_s=0.005)
        hub.tick()
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("control flow did not converge")


def _take_over_after_grant(controller: ControllerNode, handle) -> bool:
    try:
        controller.take_over(handle)
    except HandleNotGranted:
        return False
    return True


def test_complete_memory_runtime_assign_takeover_action_handover(tmp_path: Path) -> None:
    runtime = MemoryRuntime()
    fake = FakeRobot()
    hub = Hub(
        HubConfig(management_endpoint="memory://hub", database_path=tmp_path / "hub.sqlite3"),
        runtime=runtime,
    )
    controller = ControllerNode(
        ControllerNodeConfig(
            node_id_path=_write_id(tmp_path / "controller-id", "fake-controller"),
            display_name="Fake Controller",
            action_schemas=("test.float.v1",),
            action_endpoint="memory://direct-actions",
            hub_seed="memory://hub",
        ),
        runtime=runtime,
    )
    robot = RobotNode(
        fake,
        FloatProcessor(),
        RobotNodeConfig(
            node_id_path=_write_id(tmp_path / "robot-id", "fake-robot"),
            display_name="Fake Robot",
            accepted_payload_schemas=("test.float.v1",),
            hub_seed="memory://hub",
        ),
        runtime=runtime,
    )

    try:
        hub.start()
        robot.start()
        controller.start()
        _pump(hub, lambda: len(hub.list_nodes()) == 2)

        handle = hub.assign("fake-robot", "fake-controller")
        _pump(hub, lambda: _take_over_after_grant(controller, handle))
        _pump(hub, lambda: hub.get_snapshot(handle.handle_id).handle_state is HandleState.ACTIVE)

        assert controller.publish(
            b"1.0",
            captured_monotonic_ns=time.monotonic_ns(),
            captured_utc_ns=time.time_ns(),
        )
        robot.run_cycle()
        assert fake.sent == [{"x": 1.0}]

        controller.hand_over(handle)
        _pump(hub, lambda: hub.get_snapshot(handle.handle_id).handle_state is HandleState.RELEASED)
        assert robot.control_state is RobotControlState.HOLD
    finally:
        controller.stop()
        robot.stop()
        hub.stop()
        runtime.close()
