"""Standard Isaac Teleop teleoperator package.

Public API is the ``IsaacTeleop`` / ``IsaacTeleopConfig`` pair; the live
debug dashboard lives in :mod:`.debug`.
"""

from .isaac_teleop import IsaacTeleop, IsaacTeleopConfig

__all__ = ["IsaacTeleop", "IsaacTeleopConfig"]
