import sys
import types

from .mock_robot import (
    MockCamera,
    MockMotor,
    MockRobot,
    MockRobotConfig,
)

# ── sys.modules shim for the lerobot factory ────────────────────────────────
#
# ``lerobot.robots.utils.make_robot_from_config`` has a ``mock_robot`` branch
# that does ``from tests.mocks.mock_robot import MockRobot``. The upstream
# ``tests/`` package ships empty (draccus's test directory; no ``mocks/``
# subpackage), so that import would raise ``ModuleNotFoundError`` and abort
# callers that go through ``lerobot.robots.make_robot_from_config`` (e.g.
# ``examples.isaac_teleop_to_so101.common``). To avoid patching vendored lerobot
# code, we register our :class:`MockRobot` class at the upstream-expected
# import path on first import of this module.
#
# Safe because the upstream ``tests/`` package ships empty; injecting
# ``tests.mocks`` and ``tests.mocks.mock_robot`` adds two namespaces that no
# other code in the venv references. ``tests`` itself does not need to be
# pre-loaded -- Python's import system resolves it lazily from ``sys.path``
# when our ``tests.mocks`` entry is first accessed.
_mocks_mod: types.ModuleType = sys.modules.setdefault("tests.mocks", types.ModuleType("tests.mocks"))
_mocks_mod.__path__ = []  # mark as a package so submodule imports resolve
_shim: types.ModuleType = sys.modules.setdefault(
    "tests.mocks.mock_robot", types.ModuleType("tests.mocks.mock_robot")
)
_shim.MockRobot = MockRobot


__all__ = [
    "MockCamera",
    "MockMotor",
    "MockRobot",
    "MockRobotConfig",
]
