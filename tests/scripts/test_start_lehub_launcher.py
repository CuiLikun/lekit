from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "src/lekit/scripts/start_lehub.sh"


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_launcher_starts_all_nodes_and_stops_the_group_when_one_exits(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"
    _executable(
        fake_bin / "ip",
        """#!/usr/bin/env bash
if [[ "$*" == *"route get"* ]]; then
  echo "1.1.1.1 via 192.168.5.1 dev wlan0 src 192.168.5.24"
else
  echo "1: can0: <NOARP,UP,LOWER_UP> mtu 16 state UNKNOWN"
fi
""",
    )
    _executable(fake_bin / "ss", "#!/usr/bin/env bash\nexit 0\n")
    _executable(fake_bin / "curl", "#!/usr/bin/env bash\nexit 0\n")
    _executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
echo "$*" >> "$LEHUB_TEST_COMMAND_LOG"
if [[ "$*" == *"lekit robot"* ]]; then
  exit 7
fi
trap 'exit 0' TERM INT
while true; do sleep 0.1; done
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LEHUB_TEST_COMMAND_LOG": str(command_log),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 7
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("run lekit hub ") for line in commands)
    assert any(line.startswith("run lekit teleop ") for line in commands)
    robot_command = next(line for line in commands if line.startswith("run lekit robot "))
    assert "--control-rate-hz 30" in robot_command
    assert "--processor.translation_scale 1.0" in robot_command
    assert "--processor.rotation_scale 0.1" in robot_command
    assert "--processor.max_translation_from_anchor_m 0.10" in robot_command
    assert "--processor.max_rotation_from_anchor_rad 0.17453292519943295" in robot_command
    assert "192.168.5.24" in result.stdout
