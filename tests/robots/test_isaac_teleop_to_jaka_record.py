import ast
import csv
import importlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


def test_gripper_toggle_preserves_target_in_disengaged_hold_frames(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")

    class FakeRobot:
        action_features = ["gripper.pos"]

        def is_in_servo(self):
            return False

        def send_action(self, action, *, use_servo=True):
            return dict(action)

    observation = {"gripper.pos": 0.0}
    hold = record_module.HoldLatch(["gripper.pos"])
    toggle = record_module.TriggerGripperToggle()
    recorded = []
    for trigger in (1.0, 0.0, 1.0, 0.0):
        action = toggle.apply(hold.resolve(None, observation), observation, trigger)
        sent_action = record_module._send_action_for_clutch_state(
            FakeRobot(),
            action,
            engaged=False,
            observation=observation,
            gripper_target=toggle.position,
        )
        recorded.append(sent_action["gripper.pos"])

    assert recorded == [1.0, 1.0, 0.0, 0.0]


def test_controller_buttons_emit_one_a_edge_and_one_b_long_press(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    buttons = record_module.ControllerButtons(reset_hold_s=1.0)

    assert buttons.update(0.0, 0.0, 10.0) == (False, False)
    assert buttons.update(1.0, 0.0, 10.1) == (True, False)
    assert buttons.update(1.0, 0.0, 10.2) == (False, False)
    assert buttons.update(0.0, 1.0, 11.0) == (False, False)
    assert buttons.update(0.0, 1.0, 11.9) == (False, False)
    assert buttons.update(0.0, 1.0, 12.0) == (False, True)
    assert buttons.update(0.0, 1.0, 13.0) == (False, False)
    assert buttons.update(0.0, 0.0, 13.1) == (False, False)
    assert buttons.update(0.0, 1.0, 14.0) == (False, False)
    assert buttons.update(0.0, 1.0, 15.0) == (False, True)

    tracking_safe = record_module.ControllerButtons(reset_hold_s=1.0)
    assert tracking_safe.update(1.0, 0.0, 20.0) == (True, False)
    assert tracking_safe.update(0.0, 0.0, 20.1, tracking=False) == (False, False)
    assert tracking_safe.update(1.0, 0.0, 20.2) == (False, False)


def test_episode_controller_starts_ready_and_tracks_recording_time(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    episode = record_module.EpisodeController()

    assert episode.state == "ready"
    assert episode.elapsed_s(10.0) == 0.0
    assert episode.toggle_recording(10.0) is False
    assert episode.state == "recording"
    assert episode.elapsed_s(12.5) == pytest.approx(2.5)
    assert episode.toggle_recording(13.0) is True
    assert episode.state == "ready"
    assert episode.elapsed_s(14.0) == 0.0


def test_robot_reset_exits_servo_and_uses_exact_planned_joint_action(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    calls: list[tuple] = []
    reset_joints = [-0.956, 1.903, 1.427, 1.368, -1.590, -0.290]

    class FakeRobot:
        config = SimpleNamespace(reset_joints=reset_joints, max_relative_target=0.05)

        def is_in_servo(self):
            return True

        def servo_enable(self, enabled):
            calls.append(("servo_enable", enabled))

        def send_action(self, action, *, use_servo=True):
            calls.append(("send_action", dict(action), use_servo, self.config.max_relative_target))
            return dict(action)

    robot = FakeRobot()
    result = record_module._reset_robot(robot)
    expected = {f"joint_{index}.pos": value for index, value in enumerate(reset_joints, start=1)}

    assert result == expected
    assert calls == [
        ("servo_enable", False),
        ("send_action", expected, False, None),
    ]
    assert robot.config.max_relative_target == 0.05


def test_record_loop_only_buffers_frames_between_a_press_edges(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    schema_robot = JakaRobot(JakaRobotConfig(ip="127.0.0.1"))
    features = build_dataset_features(schema_robot, use_videos=False)
    observation = {
        key: float(index) / 10
        for index, key in enumerate(schema_robot.observation_features, start=1)
        if isinstance(schema_robot.observation_features[key], type)
    }

    class FakeRobot:
        action_features = schema_robot.action_features

        def get_observation(self):
            return dict(observation)

        def is_in_servo(self):
            return False

        def send_action(self, action, *, use_servo=True):
            return dict(action)

    class FakeDevice:
        def __init__(self):
            self.telemetry = {}
            self._a_values = iter((0.0, 1.0, 0.0, 1.0))

        def compute(self, _obs):
            self.telemetry.update(
                a_button=next(self._a_values),
                b_button=0.0,
                trigger=0.0,
                clutch_engaged=False,
            )
            return None

        def rearm(self):
            pass

    class FakeDataset:
        def __init__(self):
            self.features = features
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

        def has_pending_frames(self):
            return bool(self.frames)

        def clear_episode_buffer(self):
            self.frames.clear()

    class FakeLive:
        def update(self, *_args, **_kwargs):
            pass

        def refresh(self):
            pass

    class FakeRerun:
        def __init__(self):
            self.switches = 0
            self.frames = []

        def switch_record(self):
            self.switches += 1

        def log(self, frame):
            self.frames.append(frame)

    monkeypatch.setattr(record_module, "_jaka_status", lambda _robot: {})
    monkeypatch.setattr(record_module, "precise_sleep", lambda _duration: None)
    monkeypatch.setattr(
        record_module.logging,
        "info",
        lambda *_args, **_kwargs: pytest.fail(
            "button-state logging bypasses Rich Live and leaves terminal border fragments"
        ),
    )
    dataset = FakeDataset()
    rerun = FakeRerun()
    events = {
        "exit_early": False,
        "toggle_recording": False,
        "reset_robot": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    outcome = record_module._record_loop(
        FakeRobot(),
        FakeDevice(),
        [key for key in schema_robot.action_features if key.startswith("ee.")]
        + ["gripper.pos"],
        events,
        30,
        FakeLive(),
        dataset=dataset,
        control_time_s=60,
        single_task="test",
        episode_number=2,
        rerun_logger=rerun,
    )

    assert outcome == "completed"
    assert len(dataset.frames) == 2
    assert rerun.switches == 1
    assert [frame["framestep"] for frame in rerun.frames] == [0, 1]
    assert all("joint_1.pos" in frame for frame in rerun.frames)
    assert all("gripper.pos" in frame for frame in rerun.frames)
    assert all(frame["task"] == "test" for frame in rerun.frames)
    assert all(frame["episode_number"] == 2 for frame in rerun.frames)


def test_rerun_logger_detects_pos_fields_and_logs_direct_curves(monkeypatch):
    rerun_module = importlib.import_module("src.utils.rerun_utils")
    logged: list[tuple[str, float]] = []

    class FakeQueue:
        def put_nowait(self, _item):
            pass

    logger = rerun_module.RerunLogger.__new__(rerun_module.RerunLogger)
    logger._rec = object()
    logger._joint_count = None
    logger._position_keys = []
    logger._camera_slots = []
    logger._image_keys = []
    logger._next_frame_seq = 0
    logger._blueprint_sent = True
    logger._queue = FakeQueue()

    monkeypatch.setattr(rerun_module.rr, "set_time", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rerun_module.rr, "Scalars", float)
    monkeypatch.setattr(
        rerun_module.rr,
        "log",
        lambda path, value, **_kwargs: logged.append((path, value)),
    )

    logger.log({"joint_1.pos": 0.1, "joint_2.pos": 0.2, "gripper.pos": 0.25})
    logger._log_sync({"joint_1.pos": 0.1, "joint_2.pos": 0.2, "gripper.pos": 0.25})

    assert logger._position_keys == ["joint_1.pos", "joint_2.pos", "gripper.pos"]
    assert ("joint_1.pos", 0.1) in logged
    assert ("gripper.pos", 0.25) in logged


def test_rerun_logger_draws_episode_number_large_at_image_top_center():
    rerun_module = importlib.import_module("src.utils.rerun_utils")
    logger = rerun_module.RerunLogger.__new__(rerun_module.RerunLogger)
    image = np.zeros((100, 300, 3), dtype=np.uint8)

    result = logger._draw_episode_label(image, 4)

    assert result is not image
    ys, xs = np.where(np.any(result != 0, axis=2))
    assert ys.min() <= logger._EPISODE_TOP_MARGIN_PX + 2
    assert ys.max() < 75
    assert (xs.min() + xs.max()) / 2 == pytest.approx(150.0, abs=2.0)
    assert np.count_nonzero(result) > 0


def test_record_loop_skips_timed_out_camera_frame_without_stopping_control(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    schema_robot = JakaRobot(JakaRobotConfig(ip="127.0.0.1"))
    features = build_dataset_features(schema_robot, use_videos=False)
    observation = {
        key: float(index) / 10
        for index, key in enumerate(schema_robot.observation_features, start=1)
        if isinstance(schema_robot.observation_features[key], type)
    }

    class FakeRobot:
        action_features = schema_robot.action_features

        def __init__(self):
            self.read_count = 0
            self.send_count = 0
            self.servo_on = False

        def get_observation(self):
            self.read_count += 1
            if self.read_count == 2:
                raise record_module.JakaCameraTimeoutError(
                    "hand",
                    observation,
                    TimeoutError("latest frame is too old"),
                )
            return dict(observation)

        def is_in_servo(self):
            return self.servo_on

        def servo_enable(self, enabled, *, representation="joints"):
            self.servo_on = enabled

        def send_action(self, action, *, use_servo=True):
            self.send_count += 1
            sent = {key: observation[key] for key in self.action_features if key in observation}
            sent.update(action)
            return sent

    class FakeDevice:
        def __init__(self):
            self.telemetry = {}
            self._a_values = iter((1.0, 0.0, 0.0, 1.0))
            self.compute_count = 0

        def compute(self, obs):
            assert "ee.x" in obs
            self.compute_count += 1
            self.telemetry.update(
                a_button=next(self._a_values),
                b_button=0.0,
                trigger=0.0,
                clutch_engaged=True,
            )
            return {
                key: obs[key]
                for key in ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            }

        def rearm(self):
            pass

    class FakeDataset:
        def __init__(self):
            self.features = features
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

        def has_pending_frames(self):
            return bool(self.frames)

        def clear_episode_buffer(self):
            self.frames.clear()

    class FakeLive:
        def update(self, *_args, **_kwargs):
            pass

        def refresh(self):
            pass

    monkeypatch.setattr(record_module, "_jaka_status", lambda _robot: {})
    monkeypatch.setattr(record_module, "precise_sleep", lambda _duration: None)
    robot = FakeRobot()
    device = FakeDevice()
    dataset = FakeDataset()
    events = {
        "exit_early": False,
        "toggle_recording": False,
        "reset_robot": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    outcome = record_module._record_loop(
        robot,
        device,
        [key for key in schema_robot.action_features if key.startswith("ee.")]
        + ["gripper.pos"],
        events,
        30,
        FakeLive(),
        dataset=dataset,
        control_time_s=60,
        single_task="test",
    )

    assert outcome == "completed"
    assert robot.read_count == 4
    assert robot.send_count == 4
    assert device.compute_count == 4
    assert len(dataset.frames) == 2


def test_episode_save_keeps_live_active_while_suppressing_progress(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    calls: list[str] = []

    class FakeLive:
        is_started = True

        def stop(self):
            pytest.fail("stopping a tall Live panel leaves partial rows in terminal scrollback")

        def start(self, *, refresh=False):
            pytest.fail("the Live panel must not be restarted during episode saves")

    class ProgressPrintingDataset:
        def save_episode(self):
            assert live.is_started
            calls.append("dataset.save_episode")

    live = FakeLive()
    record_module._save_episode_quietly(ProgressPrintingDataset())

    assert calls == ["dataset.save_episode"]


def test_stable_live_clears_dynamic_render_before_printing_final_panel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    calls: list[tuple] = []

    class FakeConsole:
        def print(self, renderable):
            calls.append(("print", renderable))

    class FakeLive:
        def __init__(self, renderable, **kwargs):
            assert kwargs["transient"] is True
            assert kwargs["screen"] is False
            assert kwargs["auto_refresh"] is False
            self.renderable = renderable
            self.console = FakeConsole()

        def start(self):
            calls.append(("start",))

        def update(self, renderable, *, refresh=False):
            self.renderable = renderable
            calls.append(("update", getattr(renderable, "plain", renderable), refresh))

        def stop(self):
            calls.append(("stop",))

    monkeypatch.setattr(record_module, "Live", FakeLive)

    with record_module._stable_live("initial") as live:
        live.renderable = "final panel"

    assert calls == [
        ("start",),
        ("update", "", False),
        ("stop",),
        ("print", "final panel"),
    ]


def test_hold_latch_captures_measured_pose_on_engage_release(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    latch = record_module.HoldLatch(["ee.x", "ee.y", "ee.z"])
    measured = {"ee.x": 0.10, "ee.y": 0.20, "ee.z": 0.30}
    commanded = {"ee.x": 0.30, "ee.y": 0.20, "ee.z": 0.30}

    assert latch.resolve(None, measured) == measured
    assert latch.resolve(commanded, measured) == commanded
    # Releasing the deadman cancels any pending controller trajectory at the
    # measured pose instead of continuing toward the old hand target.
    lagging_feedback = {"ee.x": 0.24, "ee.y": 0.20, "ee.z": 0.30}
    assert latch.resolve(None, lagging_feedback) == lagging_feedback


def test_loop_rate_monitor_measures_a_rolling_wall_clock_window(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    monitor = record_module.LoopRateMonitor(window_s=1.0)

    assert monitor.update(10.0) is None
    for index in range(1, 11):
        measured = monitor.update(10.0 + index * 0.1)
    assert measured == pytest.approx(10.0)

    # Old samples leave the rolling window instead of diluting the current rate.
    assert monitor.update(11.5) == pytest.approx(6.0)
    assert monitor.update(12.0) == pytest.approx(2.0)


def test_arm_servo_follows_clutch_lifecycle_and_preserves_hold_action(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    calls: list[tuple] = []
    schema_robot = JakaRobot(JakaRobotConfig(ip="127.0.0.1"))
    robot_action_features = schema_robot.action_features
    dataset_features = build_dataset_features(schema_robot, use_videos=False)

    class FakeRobot:
        servo_active = False
        action_features = robot_action_features

        def is_in_servo(self):
            return self.servo_active

        def servo_enable(self, enabled, *, representation="joints"):
            calls.append(("servo_enable", enabled, representation))
            self.servo_active = enabled

        def send_action(self, action):
            calls.append(("send_action", dict(action)))
            return dict(action)

    robot = FakeRobot()
    action = {
        "ee.x": 0.1,
        "ee.y": 0.2,
        "ee.z": 0.3,
        "ee.roll": 0.0,
        "ee.pitch": 0.0,
        "ee.yaw": 0.0,
        "gripper.pos": 1.0,
    }
    observation = {
        key: index / 10 for index, key in enumerate(robot_action_features, start=1)
    }

    assert record_module._send_action_for_clutch_state(robot, action, engaged=True) == action
    assert calls == [
        ("servo_enable", True, "eef"),
        ("send_action", action),
    ]

    calls.clear()
    held = record_module._send_action_for_clutch_state(
        robot, action, engaged=False, observation=observation
    )
    expected_hold = {**observation, **action}
    assert held == expected_hold
    record_module.build_dataset_frame(dataset_features, held, prefix=record_module.ACTION)
    assert calls == [
        ("servo_enable", False, "joints"),
        ("send_action", {"gripper.pos": 1.0}),
    ]

    calls.clear()
    hold_without_gripper = {key: value for key, value in action.items() if key != "gripper.pos"}
    assert (
        record_module._send_action_for_clutch_state(
            robot, hold_without_gripper, engaged=False, observation=observation
        )
        == {**observation, **hold_without_gripper}
    )
    assert calls == []


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


def test_control_panel_calls_use_compact_signature():
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
    assert all(len(call.args) == 4 for call in calls)


def test_record_does_not_teardown_hardware_from_keyboard_thread():
    source_path = Path(__file__).parents[2] / "examples/isaac_teleop_to_jaka/record.py"
    tree = ast.parse(source_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "init_keyboard_listener"
    ]

    assert len(calls) == 1
    assert calls[0].args == []
    assert calls[0].keywords == []


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
            {
                "clutch_engaged": True,
                "head_is_tracking": True,
                "grip_pos": (0.1, 0.2, 0.3),
                "grip_quat": (0.0, 0.0, 0.0, 1.0),
                "squeeze": 1.0,
                "trigger": 0.5,
            },
            {
                "ee.x": 0.1,
                "ee.y": 0.2,
                "ee.z": 0.3,
                "ee.roll": 0.1,
                "ee.pitch": 0.2,
                "ee.yaw": 0.3,
            },
            {
                "ee.x": 0.08,
                "ee.y": 0.23,
                "ee.z": 0.31,
                "ee.roll": 0.1,
                "ee.pitch": 0.2,
                "ee.yaw": 0.3,
                "gripper.pos": 1.0,
                "joint_1.pos": 0.12349,
                "joint_2.pos": -1.23456,
                "joint_3.pos": 2.34567,
                "joint_4.pos": -3.14159,
                "joint_5.pos": 0.0004,
                "joint_6.pos": -0.0006,
            },
            {
                "powered_on": True,
                "enabled": True,
                "servo_on": True,
                "servo_rate_hz": 100.0,
                "control_rate_hz": 29.8,
                "control_target_hz": 30.0,
                "episode_number": 2,
                "episode_total": 5,
                "episode_elapsed_s": 12.36,
                "recording": True,
            },
        )
    )

    rendered = output.getvalue()
    assert "JAKA Teleop" in rendered
    assert "Mode" not in rendered
    assert "RECORDING" in rendered
    assert "ENGAGED" in rendered
    assert "Episode" in rendered and "2 / 5   12.4 s" in rendered
    assert "A/n rec" in rendered and "stick XY pitch/yaw" in rendered
    assert "click+X roll" in rendered
    controls_line = next(line for line in rendered.splitlines() if "Controls" in line)
    assert "stick XY pitch/yaw" in controls_line and "click+X roll" in controls_line
    assert "TCP(m/rad)" in rendered
    assert "[0.080, 0.230, 0.310, 0.100, 0.200, 0.300]" in rendered
    assert "Joint(rad)" in rendered
    assert "[0.123, -1.235, 2.346, -3.142, 0.000, -0.001]" in rendered
    assert "Grip" in rendered
    assert "Error" in rendered
    assert "LOOP" in rendered and "29.8 / 30.0 Hz" in rendered
    assert "X -20.0 mm" in rendered
    assert "Y +30.0 mm" in rendered
    assert "Z +10.0 mm" in rendered
    assert "XYZ movement" not in rendered
    assert "grip_pos" not in rendered
    assert "grip_quat" not in rendered
    assert "squeeze" not in rendered
    assert "trigger" not in rendered
    assert "orientation" not in rendered
    assert "panel:" not in rendered
    assert "frame:" not in rendered
    assert "█" in rendered


def test_control_panel_deadbands_submillimetre_feedback_noise(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)

    console.print(
        record_module._control_panel(
            {},
            {"ee.x": 0.1, "ee.y": 0.2, "ee.z": 0.3},
            {"ee.x": 0.1001, "ee.y": 0.1999, "ee.z": 0.30015},
            {},
        )
    )

    rendered = output.getvalue()
    assert "X +0.0 mm" in rendered
    assert "Y +0.0 mm" in rendered
    assert "Z +0.0 mm" in rendered


@pytest.mark.parametrize(
    ("state", "border_style"),
    [("ready", "cyan"), ("recording", "green"), ("resetting", "yellow")],
)
def test_control_panel_shows_recorder_state(monkeypatch, state, border_style):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)

    panel = record_module._control_panel(
        {},
        {},
        {},
        {"record_state": state},
    )
    console.print(panel)

    assert state.upper() in output.getvalue()
    assert panel.border_style == border_style


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

    assert line.plain == "POWER ON   ENABLED OFF   SERVO ON"


def test_jaka_status_line_reports_camera_timeout_count():
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")

    line = record_module._jaka_status_line(
        {
            "powered_on": True,
            "enabled": True,
            "servo_on": False,
            "camera_timeout_count": 3,
        }
    )

    assert line.plain.endswith("CAM TIMEOUT 3")


def test_control_trace_records_disengaged_hold_and_servo_targets(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    trace_module = importlib.import_module("examples.isaac_teleop_to_jaka.control_trace")
    path = tmp_path / "control.csv"
    target = {
        "ee.x": 0.4,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.roll": 0.1,
        "ee.pitch": -0.2,
        "ee.yaw": 0.3,
    }

    with trace_module.ControlTraceWriter(path, flush_every=1) as trace:
        for actual_x in (0.4001, 0.3999):
            trace.write_frame(
                phase="record",
                raw_action=None,
                action=target,
                sent_action=target,
                observation={**target, "ee.x": actual_x},
                telemetry={
                    "clutch_engaged": False,
                    "clutch_released": True,
                    "squeeze": 0.0,
                    "trigger": 0.0,
                    "thumbstick_x": -0.25,
                    "thumbstick_y": 0.75,
                    "thumbstick_click": 1.0,
                    "grip_pos": (0.1, 0.2, 0.3),
                    "raw_grip_pos": (0.4, 0.5, 0.6),
                    "head_quat": (0.0, 0.5, 0.0, 0.8660254),
                    "head_is_tracking": True,
                    "control_yaw_deg": 60.0,
                },
                servo_status={
                    "active": True,
                    "worker_alive": True,
                    "representation": "eef",
                    "filter_mode": "none",
                    "target": tuple(target.values()),
                    "commanded_position": tuple(target.values()),
                    "send_rate_hz": 125.0,
                },
                frame_ms=8.0,
                control_rate_hz=29.5,
                control_target_hz=30.0,
            )

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2
    assert rows[1]["action_source"] == "hold"
    assert rows[1]["clutch_engaged"] == "False"
    assert rows[1]["clutch_released"] == "True"
    assert float(rows[1]["control_rate_hz"]) == pytest.approx(29.5)
    assert float(rows[1]["control_target_hz"]) == pytest.approx(30.0)
    assert float(rows[1]["target_step_norm_m"]) == 0.0
    assert float(rows[1]["tracking_error_x"]) == pytest.approx(0.0001)
    assert float(rows[1]["servo_target_x"]) == pytest.approx(0.4)
    assert float(rows[1]["servo_commanded_x"]) == pytest.approx(0.4)
    assert rows[1]["servo_filter_mode"] == "none"
    assert [float(rows[1][f"grip_{axis}_m"]) for axis in ("x", "y", "z")] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert [
        float(rows[1][f"raw_grip_{axis}_m"]) for axis in ("x", "y", "z")
    ] == pytest.approx([0.4, 0.5, 0.6])
    assert rows[1]["head_is_tracking"] == "True"
    assert [float(rows[1][f"head_quat_{axis}"]) for axis in ("x", "y", "z", "w")] == pytest.approx(
        [0.0, 0.5, 0.0, 0.8660254]
    )
    assert float(rows[1]["control_yaw_deg"]) == pytest.approx(60.0)
    assert float(rows[1]["thumbstick_x"]) == pytest.approx(-0.25)
    assert float(rows[1]["thumbstick_y"]) == pytest.approx(0.75)
    assert rows[1]["thumbstick_click"] == "1.0"


def test_build_device_isolates_feedback_and_defers_servo_start(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")

    observed_config: dict[str, object] = {}

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
        observed_config["servo_filter_mode"] = config.servo_filter_mode
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
            servo_filter_mode="none",
            user_frame_id=0,
        ),
        teleop=SimpleNamespace(cloudxr_env_file=None),
    )

    robot, _device = record_module.build_device(cfg)

    assert robot.connected
    assert observed_config == {
        "auto_enable_servo": False,
        "separate_feedback_connection": True,
        "servo_filter_mode": "cartesian_nlf",
    }
    assert startup_calls == [True]


def test_build_device_applies_xr_servo_profile(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    cfg = SimpleNamespace(
        robot=SimpleNamespace(
            auto_enable_servo=True,
            separate_feedback_connection=False,
            servo_filter_mode="none",
            user_frame_id=0,
        ),
        teleop=SimpleNamespace(
            cloudxr_env_file=None,
            servo_linear_velocity_m_s=0.15,
            servo_linear_acceleration_m_s2=0.8,
            servo_linear_jerk_m_s3=8.0,
            servo_angular_velocity_rad_s=1.0,
            servo_angular_acceleration_rad_s2=2.0,
            servo_angular_jerk_rad_s3=20.0,
        ),
    )
    observed: dict[str, float] = {}

    class FakeRobot:
        name = "jaka_robot"
        is_connected = False

        def __init__(self):
            self.config = SimpleNamespace(user_frame_id=0)

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

    def make_robot(config):
        for key in (
            "servo_eef_max_velocity_m_s",
            "servo_eef_max_acceleration_m_s2",
            "servo_filter_eef_max_jerk_m_s3",
            "servo_eef_max_angular_velocity_rad_s",
            "servo_eef_max_angular_acceleration_rad_s2",
            "servo_filter_eef_max_angular_jerk_rad_s3",
        ):
            observed[key] = getattr(config, key)
        return FakeRobot()

    monkeypatch.setattr(record_module, "JakaRobot", FakeRobot)
    monkeypatch.setattr(record_module, "make_robot_from_config", make_robot)
    monkeypatch.setattr(
        record_module,
        "make_xr_device",
        lambda _robot, _teleop: {
            "compute": lambda _obs: None,
            "startup": lambda: None,
            "cleanup": lambda: None,
            "telemetry": {},
        },
    )
    record_module.build_device(cfg)

    assert observed == {
        "servo_eef_max_velocity_m_s": 0.15,
        "servo_eef_max_acceleration_m_s2": 0.8,
        "servo_filter_eef_max_jerk_m_s3": 8.0,
        "servo_eef_max_angular_velocity_rad_s": 1.0,
        "servo_eef_max_angular_acceleration_rad_s2": 2.0,
        "servo_filter_eef_max_angular_jerk_rad_s3": 20.0,
    }


def test_build_device_rejects_non_cartesian_servo_filter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    record_module = importlib.import_module("examples.isaac_teleop_to_jaka.record")
    cfg = SimpleNamespace(
        robot=SimpleNamespace(
            auto_enable_servo=True,
            separate_feedback_connection=False,
            servo_filter_mode="joint_lpf",
        ),
        teleop=SimpleNamespace(cloudxr_env_file=None),
    )
    monkeypatch.setattr(
        record_module,
        "make_robot_from_config",
        lambda _config: pytest.fail("invalid filter must fail before connecting"),
    )

    with pytest.raises(ValueError, match="requires cartesian_nlf"):
        record_module.build_device(cfg)


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


def test_escape_keyboard_dispatch_invokes_stop_callback(monkeypatch, capsys):
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
    assert capsys.readouterr().out == ""


def test_keyboard_dispatch_maps_episode_and_reset_redundancy(monkeypatch):
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
    _, events = record_module.init_keyboard_listener(lambda: None)
    assert captured_dispatch is not None

    captured_dispatch("a")
    assert events["toggle_recording"] is True
    captured_dispatch("b")
    assert events["reset_robot"] is True
    captured_dispatch("left")
    assert events["rerecord_episode"] is True
