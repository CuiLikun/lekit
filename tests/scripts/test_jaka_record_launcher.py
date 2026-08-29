from __future__ import annotations

import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

from camera_stream.config import load_config

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "record.sh"
CAMERA_CONFIG = PROJECT_ROOT / "examples/isaac_teleop_to_jaka/camera_stream.yaml"


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_local_camera_config_publishes_both_d435i_color_streams() -> None:
    config = load_config(CAMERA_CONFIG)

    assert config.endpoints.stream_pub == "tcp://127.0.0.1:5555"
    assert [camera.model_dump(mode="json") for camera in config.cameras] == [
        {
            "name": "base_camera",
            "driver": "realsense",
            "device": {"path": None, "serial": "347522072196"},
            "profile": {"width": 640, "height": 480, "fps": 30},
            "encoding": {"codec": "jpeg", "jpeg_quality": 85},
        },
        {
            "name": "hand_camera",
            "driver": "realsense",
            "device": {"path": None, "serial": "342522070741"},
            "profile": {"width": 640, "height": 480, "fps": 30},
            "encoding": {"codec": "jpeg", "jpeg_quality": 85},
        },
    ]


def test_launcher_records_scoop_powder_into_a_timestamped_project_dataset(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    recorder_command = tmp_path / "recorder.command"
    _executable(fake_bin / "date", "#!/usr/bin/env bash\necho 20260829_170000\n")
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [[ "$*" == "run camera-stream topic list "* ]]; then
  exit 0
fi
if [[ "$*" == "run camera-stream topic echo "* ]]; then
  exit 0
fi
if [[ "$*" == "run python -m examples.isaac_teleop_to_jaka.record "* ]]; then
  printf '%s\n' "$*" > "$JAKA_RECORD_TEST_RECORDER_COMMAND"
  exit 0
fi
exit 21
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "JAKA_RECORD_TEST_RECORDER_COMMAND": str(recorder_command),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = recorder_command.read_text(encoding="utf-8")
    assert "--dataset.repo_id=sorel/scoop-powder" in command
    assert (
        f"--dataset.root={PROJECT_ROOT}/datasets/scoop-powder_20260829_170000"
        in command
    )
    assert "--dataset.single_task=scoop powder from the container" in command
    assert "--teleop.tool_tip_offset_m=[0.0,0.0,0.27]" in command


def test_launcher_starts_local_camera_server_before_recording_and_stops_it_afterward(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"
    server_pid = tmp_path / "server.pid"
    base_frame = tmp_path / "base.frame"
    hand_frame = tmp_path / "hand.frame"
    recorder_started = tmp_path / "recorder.started"
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
echo "$*" >> "$JAKA_RECORD_TEST_COMMAND_LOG"
if [[ "$*" == "run camera-stream topic list --endpoint tcp://127.0.0.1:5555 --timeout 0.5" ]]; then
  [[ -s "$JAKA_RECORD_TEST_SERVER_PID" ]]
  exit $?
fi
if [[ "$*" == "run camera-stream server --config "*"camera_stream.yaml" ]]; then
  echo "$$" > "$JAKA_RECORD_TEST_SERVER_PID"
  trap 'exit 0' TERM INT
  while true; do sleep 0.05; done
fi
if [[ "$*" == "run camera-stream topic echo base_camera/color --endpoint tcp://127.0.0.1:5555 --count 1" ]]; then
  touch "$JAKA_RECORD_TEST_BASE_FRAME"
  exit 0
fi
if [[ "$*" == "run camera-stream topic echo hand_camera/color --endpoint tcp://127.0.0.1:5555 --count 1" ]]; then
  touch "$JAKA_RECORD_TEST_HAND_FRAME"
  exit 0
fi
if [[ "$*" == "run python -m examples.isaac_teleop_to_jaka.record "* ]]; then
  [[ -s "$JAKA_RECORD_TEST_SERVER_PID" ]] || exit 20
  [[ -f "$JAKA_RECORD_TEST_BASE_FRAME" ]] || exit 22
  [[ -f "$JAKA_RECORD_TEST_HAND_FRAME" ]] || exit 23
  touch "$JAKA_RECORD_TEST_RECORDER_STARTED"
  exit 0
fi
exit 21
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "JAKA_RECORD_TEST_COMMAND_LOG": str(command_log),
        "JAKA_RECORD_TEST_SERVER_PID": str(server_pid),
        "JAKA_RECORD_TEST_BASE_FRAME": str(base_frame),
        "JAKA_RECORD_TEST_HAND_FRAME": str(hand_frame),
        "JAKA_RECORD_TEST_RECORDER_STARTED": str(recorder_started),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert recorder_started.is_file()
    commands = command_log.read_text(encoding="utf-8").splitlines()
    server_index = next(index for index, command in enumerate(commands) if "camera-stream server" in command)
    recorder_index = next(
        index
        for index, command in enumerate(commands)
        if "python -m examples.isaac_teleop_to_jaka.record" in command
    )
    assert server_index < recorder_index
    pid = int(server_pid.read_text(encoding="utf-8"))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError(f"camera-stream server process {pid} survived recorder exit")


def test_launcher_fails_when_camera_server_exits_before_becoming_ready(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    recorder_started = tmp_path / "recorder.started"
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [[ "$*" == "run camera-stream topic list "* ]]; then
  exit 1
fi
if [[ "$*" == "run camera-stream server "* ]]; then
  exit 0
fi
if [[ "$*" == "run python -m examples.isaac_teleop_to_jaka.record "* ]]; then
  touch "$JAKA_RECORD_TEST_RECORDER_STARTED"
  exit 0
fi
exit 21
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "JAKA_RECORD_TEST_RECORDER_STARTED": str(recorder_started),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert not recorder_started.exists()
    assert "camera-stream exited before becoming ready" in result.stderr


def test_launcher_force_stops_an_owned_camera_server_that_ignores_term(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    server_pid_path = tmp_path / "server.pid"
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [[ "$*" == "run camera-stream topic list "* ]]; then
  [[ -s "$JAKA_RECORD_TEST_SERVER_PID" ]]
  exit $?
fi
if [[ "$*" == "run camera-stream server "* ]]; then
  echo "$$" > "$JAKA_RECORD_TEST_SERVER_PID"
  trap '' TERM
  while true; do sleep 0.05; done
fi
if [[ "$*" == "run camera-stream topic echo "* ]]; then
  exit 0
fi
if [[ "$*" == "run python -m examples.isaac_teleop_to_jaka.record "* ]]; then
  exit 0
fi
exit 21
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "JAKA_RECORD_TEST_SERVER_PID": str(server_pid_path),
    }

    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    finished = True
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        finished = False
        process.kill()
        process.wait(timeout=1)
    finally:
        if server_pid_path.is_file():
            server_pid = int(server_pid_path.read_text(encoding="utf-8"))
            with suppress(ProcessLookupError):
                os.kill(server_pid, signal.SIGKILL)

    assert finished, "record.sh hung while stopping an unresponsive camera-stream server"


def test_launcher_reuses_an_existing_camera_server_without_stopping_it(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    server_started = tmp_path / "server.started"
    recorder_started = tmp_path / "recorder.started"
    existing_server = subprocess.Popen(["sleep", "30"])
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [[ "$*" == "run camera-stream topic list "* ]]; then
  exit 0
fi
if [[ "$*" == "run camera-stream server "* ]]; then
  touch "$JAKA_RECORD_TEST_SERVER_STARTED"
  exit 30
fi
if [[ "$*" == "run camera-stream topic echo "* ]]; then
  exit 0
fi
if [[ "$*" == "run python -m examples.isaac_teleop_to_jaka.record "* ]]; then
  touch "$JAKA_RECORD_TEST_RECORDER_STARTED"
  exit 0
fi
exit 21
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "JAKA_RECORD_TEST_SERVER_STARTED": str(server_started),
        "JAKA_RECORD_TEST_RECORDER_STARTED": str(recorder_started),
    }

    try:
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert recorder_started.is_file()
        assert not server_started.exists()
        assert existing_server.poll() is None
    finally:
        existing_server.terminate()
        existing_server.wait(timeout=1)


def test_launcher_reports_camera_status_when_a_topic_never_produces_a_frame(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    recorder_started = tmp_path / "recorder.started"
    _executable(fake_bin / "timeout", "#!/usr/bin/env bash\nexit 124\n")
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [[ "$*" == "run camera-stream topic list "* ]]; then
  exit 0
fi
if [[ "$*" == "run camera-stream topic info base_camera/color "* ]]; then
  echo '{"state":"OFFLINE","last_error":"Device or resource busy"}'
  exit 0
fi
if [[ "$*" == "run python -m examples.isaac_teleop_to_jaka.record "* ]]; then
  touch "$JAKA_RECORD_TEST_RECORDER_STARTED"
  exit 0
fi
exit 21
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "JAKA_RECORD_TEST_RECORDER_STARTED": str(recorder_started),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert not recorder_started.exists()
    assert "base_camera/color did not produce a frame" in result.stderr
    assert "Device or resource busy" in result.stderr


def test_launcher_interrupts_recording_when_its_camera_server_dies_after_readiness(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    server_pid_path = tmp_path / "server.pid"
    recorder_pid_path = tmp_path / "recorder.pid"
    release_server = tmp_path / "release-server"
    recorder_interrupted = tmp_path / "recorder.interrupted"
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [[ "$*" == "run camera-stream topic list "* ]]; then
  [[ -s "$JAKA_RECORD_TEST_SERVER_PID" ]]
  exit $?
fi
if [[ "$*" == "run camera-stream server "* ]]; then
  echo "$$" > "$JAKA_RECORD_TEST_SERVER_PID"
  while [[ ! -f "$JAKA_RECORD_TEST_RELEASE_SERVER" ]]; do sleep 0.05; done
  exit 0
fi
if [[ "$*" == "run camera-stream topic echo base_camera/color "* ]]; then
  exit 0
fi
if [[ "$*" == "run camera-stream topic echo hand_camera/color "* ]]; then
  touch "$JAKA_RECORD_TEST_RELEASE_SERVER"
  exit 0
fi
if [[ "$*" == "run python -m examples.isaac_teleop_to_jaka.record "* ]]; then
  echo "$$" > "$JAKA_RECORD_TEST_RECORDER_PID"
  trap 'touch "$JAKA_RECORD_TEST_RECORDER_INTERRUPTED"; exit 0' TERM
  while true; do sleep 0.05; done
fi
exit 21
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "JAKA_RECORD_TEST_SERVER_PID": str(server_pid_path),
        "JAKA_RECORD_TEST_RECORDER_PID": str(recorder_pid_path),
        "JAKA_RECORD_TEST_RELEASE_SERVER": str(release_server),
        "JAKA_RECORD_TEST_RECORDER_INTERRUPTED": str(recorder_interrupted),
    }

    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    finished = True
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        finished = False
        process.kill()
        process.wait(timeout=1)
    finally:
        for pid_path in (recorder_pid_path, server_pid_path):
            if pid_path.is_file():
                pid = int(pid_path.read_text(encoding="utf-8"))
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)

    assert finished, "record.sh did not notice that its camera-stream server exited"
    assert process.returncode != 0
    assert recorder_interrupted.is_file()
