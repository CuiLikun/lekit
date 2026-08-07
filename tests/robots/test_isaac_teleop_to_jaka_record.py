import ast
import importlib
import io
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from lerobot.datasets import LeRobotDatasetMetadata
from robots.jaka_robot.dataset_features import build_dataset_features
from robots.jaka_robot.jaka_robot import JakaRobot, JakaRobotConfig


def test_trigger_gripper_toggle_changes_state_only_on_press_edges(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    toggle = record_module.TriggerGripperToggle()
    observation = {"gripper.pos": 0.0}
    action = {"ee.x": 0.1, "gripper.pos": 0.4}

    assert "gripper.pos" not in toggle.apply(action, observation, 0.0)
    assert toggle.apply(action, observation, 0.8)["gripper.pos"] == 1.0
    assert "gripper.pos" not in toggle.apply(action, observation, 0.8)
    assert "gripper.pos" not in toggle.apply(action, observation, 0.0)
    assert toggle.apply(action, observation, 0.8)["gripper.pos"] == 0.0


def test_dataset_features_are_lerobot_metadata_compatible(tmp_path):
    """The recorder must convert robot descriptors to dataset feature metadata."""
    robot = JakaRobot(JakaRobotConfig(ip="127.0.0.1"))

    features = build_dataset_features(robot, use_videos=True)

    assert all("dtype" in feature for feature in features.values())
    metadata = LeRobotDatasetMetadata.create(
        repo_id="test/jaka-schema",
        fps=30,
        robot_type=robot.name,
        features=features,
        root=tmp_path / "dataset",
        use_videos=True,
    )
    assert features.items() <= metadata.features.items()


def test_control_panel_calls_pass_frame_duration():
    source_path = Path(__file__).parents[2] / "examples/isaac_teleop_to_jaka/record.py"
    tree = ast.parse(source_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_control_panel"
    ]

    assert calls
    assert all(len(call.args) == 5 for call in calls)


def test_signed_horizontal_bar_has_stable_width_and_direction(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")

    negative = record_module._signed_horizontal_bar(-0.05, 0.05, width=11).plain
    zero = record_module._signed_horizontal_bar(0.0, 0.05, width=11).plain
    positive = record_module._signed_horizontal_bar(0.05, 0.05, width=11).plain

    assert negative == "█████│─────"
    assert zero == "─────│─────"
    assert positive == "─────│█████"
    assert len(negative) == len(zero) == len(positive) == 11


def test_control_panel_visualizes_only_position_delta(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)

    console.print(
        record_module._control_panel(
            {},
            {"ee.x": 0.1, "ee.y": 0.2, "ee.z": 0.3},
            {"ee.x": 0.08, "ee.y": 0.23, "ee.z": 0.31},
            {},
            8.0,
        )
    )

    rendered = output.getvalue()
    assert "delta.position_m" in rendered
    assert "X -0.0200 m" in rendered
    assert "Y +0.0300 m" in rendered
    assert "Z +0.0100 m" in rendered
    assert "XYZ movement" not in rendered
    assert "█" in rendered


def test_jaka_status_contains_controller_and_servo_details(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")

    class FakeRobot:
        def get_controller_state(self):
            return {"powered_on": True, "enabled": True}

        def is_in_servo(self):
            return True

        def get_servo_status(self):
            return {
                "active": True,
                "worker_alive": True,
                "representation": "eef",
                "send_rate_hz": 125.34,
                "frames_sent": 42,
                "queue_depth": 3,
                "last_error": None,
            }

    status = record_module._jaka_status(FakeRobot())

    assert status == {
        "powered_on": True,
        "enabled": True,
        "servo_on": True,
        "servo_sender_active": True,
        "servo_sender_alive": True,
        "servo_representation": "eef",
        "servo_rate_hz": 125.3,
        "servo_frames_sent": 42,
        "servo_queue_depth": 3,
        "servo_last_error": None,
    }


def test_jaka_status_line_shows_only_three_color_coded_states(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")

    line = record_module._jaka_status_line(
        {"powered_on": True, "enabled": False, "servo_on": True, "servo_sender_active": True}
    )

    assert line.plain == "powered_on=True  enabled=False  servo_on=True"
    assert [span.style for span in line.spans] == ["green", "white", "green"]


def test_build_device_isolates_feedback_and_defers_servo_start(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")

    observed_config: dict[str, bool] = {}

    class FakeRobot:
        name = "jaka_robot"

        def __init__(self, config):
            self.config = config
            self.connected = False

        def connect(self):
            self.connected = True

        def disconnect(self):
            self.connected = False

    def make_robot(config):
        observed_config["auto_enable_servo"] = config.auto_enable_servo
        observed_config["separate_feedback_connection"] = config.separate_feedback_connection
        return FakeRobot(config)

    startup_calls: list[bool] = []
    monkeypatch.setattr(record_module, "JakaRobot", FakeRobot)
    monkeypatch.setattr(record_module, "make_robot_from_config", make_robot)
    monkeypatch.setattr(
        record_module,
        "make_xr_device",
        lambda _robot, _teleop: {
            "compute": lambda _obs: None,
            "startup": lambda: startup_calls.append(True),
            "cleanup": lambda: None,
            "telemetry": {},
        },
    )
    cfg = SimpleNamespace(
        robot=SimpleNamespace(
            auto_enable_servo=True,
            separate_feedback_connection=False,
            user_frame_id=0,
        ),
        teleop=SimpleNamespace(cloudxr_env_file=None),
    )

    robot, _device = record_module.build_device(cfg)

    assert robot.connected
    assert observed_config == {
        "auto_enable_servo": False,
        "separate_feedback_connection": True,
    }
    assert startup_calls == [True]


def test_hardware_stop_callback_is_idempotent_and_orders_teardown():
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    calls: list[str] = []

    class FakeDevice:
        def cleanup(self):
            calls.append("device.cleanup")

    class FakeRobot:
        is_connected = True

        def disconnect(self):
            calls.append("robot.disconnect")

    stop = record_module._make_hardware_stop_callback(FakeRobot(), FakeDevice())
    stop()
    stop()

    assert calls == ["device.cleanup", "robot.disconnect"]


def test_escape_keyboard_dispatch_invokes_stop_callback(monkeypatch):
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    captured_dispatch = None

    def fake_create_key_listener(dispatch, *, controls_help=""):
        nonlocal captured_dispatch
        captured_dispatch = dispatch
        return "listener"

    monkeypatch.setattr(
        "lerobot.utils.keyboard_input.create_key_listener",
        fake_create_key_listener,
    )
    stopped: list[bool] = []

    listener, events = record_module.init_keyboard_listener(lambda: stopped.append(True))
    assert listener == "listener"
    assert captured_dispatch is not None

    captured_dispatch("esc")

    assert events["stop_recording"] is True
    assert events["exit_early"] is True
    assert stopped == [True]
