"""Configuration for the standalone Isaac XR teleoperator."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("isaac_teleop")
@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Configure an OpenXR controller as an operator-frame input device.

    This configuration deliberately contains no robot frame, TCP, inverse
    kinematics, or actuator fields. ``get_action()`` always reports a relative
    pose in the operator frame: +X right, +Y forward, +Z up.
    """

    app_name: str = "LeTeleop"
    auto_launch_cloudxr: bool = True
    cloudxr_env_file: str | None = None
    cloudxr_install_dir: str = ".cloudxr"
    connect_timeout_s: float | None = None
    connect_retry_interval_s: float = 2.0
    squeeze_engage_threshold: float = 0.5
    squeeze_release_threshold: float = 0.3
    use_head_yaw: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.use_head_yaw, bool):
            raise ValueError("use_head_yaw must be a boolean")
        for name in ("squeeze_engage_threshold", "squeeze_release_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.squeeze_release_threshold > self.squeeze_engage_threshold:
            raise ValueError("squeeze_release_threshold must not exceed squeeze_engage_threshold")
        if not self.cloudxr_install_dir.strip():
            raise ValueError("cloudxr_install_dir must not be empty")
        if self.connect_timeout_s is not None and (
            not math.isfinite(self.connect_timeout_s) or self.connect_timeout_s <= 0.0
        ):
            raise ValueError("connect_timeout_s must be finite and positive when set")
        if not math.isfinite(self.connect_retry_interval_s) or self.connect_retry_interval_s <= 0.0:
            raise ValueError("connect_retry_interval_s must be finite and positive")
