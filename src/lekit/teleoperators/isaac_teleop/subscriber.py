"""Standard LeRobot teleoperator backed by an independent Isaac teleop node."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.errors import DeviceNotConnectedError

from .protocol import TeleopFrame, action_features, neutral_action, normalize_action
from .transport import ZmqTeleopReceiver

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("isaac_teleop_node")
@dataclass(kw_only=True)
class IsaacTeleopNodeConfig(TeleoperatorConfig):
    """Configure a latest-frame subscription to an independent teleop node."""

    endpoint: str = "tcp://127.0.0.1:5557"
    first_frame_timeout_s: float = 5.0
    stale_after_s: float = 0.25
    rearm_squeeze_threshold: float = 0.3

    @property
    def type(self) -> str:
        return "isaac_teleop_node"

    @classmethod
    def get_known_choices(cls) -> dict[str, type[IsaacTeleopNodeConfig]]:
        """Keep concrete subscriber consumers closed to direct XR configs."""

        return {"isaac_teleop_node": cls}

    @classmethod
    def get_choice_class(cls, name: str) -> type[IsaacTeleopNodeConfig]:
        if name != "isaac_teleop_node":
            raise KeyError(name)
        return cls

    @classmethod
    def get_choice_name(cls, subcls: type) -> str:
        del subcls
        raise ValueError("Decode IsaacTeleopNodeConfig through its sealed node choice")

    @classmethod
    def default_choice_name(cls) -> str:
        return "isaac_teleop_node"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError(f"endpoint has an invalid port: {self.endpoint!r}") from error
        if parsed.scheme != "tcp" or not parsed.hostname or port is None:
            raise ValueError("endpoint must be a TCP URL such as tcp://127.0.0.1:5557")
        for name in ("first_frame_timeout_s", "stale_after_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        threshold = self.rearm_squeeze_threshold
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("rearm_squeeze_threshold must be finite and in [0, 1]")


class IsaacTeleopNodeSubscriber(Teleoperator):
    """Expose a remote atomic controller stream through the LeRobot interface."""

    config_class = IsaacTeleopNodeConfig
    name = "isaac_teleop_node_subscriber"

    def __init__(
        self,
        config: IsaacTeleopNodeConfig,
        *,
        receiver_factory: Callable[[str], Any] = ZmqTeleopReceiver,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._receiver_factory = receiver_factory
        self._clock = clock
        self._receiver: Any | None = None
        self._latest_frame: TeleopFrame | None = None
        self._latest_action: RobotAction = neutral_action()
        self._last_received_at: float | None = None
        self._session_id: str | None = None
        self._last_sequence = -1
        self._armed = False

    @property
    def action_features(self) -> dict:
        return action_features()

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._receiver is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def latest_frame(self) -> TeleopFrame | None:
        """Newest accepted protocol frame, including diagnostic metadata."""

        return self._latest_frame

    def connect(self, calibrate: bool = True) -> None:
        if self._receiver is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")
        if calibrate:
            self.calibrate()
        receiver = self._receiver_factory(self.config.endpoint)
        self._receiver = receiver
        deadline = self._clock() + self.config.first_frame_timeout_s
        try:
            while True:
                remaining_s = deadline - self._clock()
                if remaining_s <= 0.0:
                    raise TimeoutError(
                        f"No first teleop frame received from {self.config.endpoint} "
                        f"within {self.config.first_frame_timeout_s:g} seconds"
                    )
                try:
                    frame = receiver.receive_latest(timeout_s=min(0.05, remaining_s))
                except ValueError:
                    frame = None
                if frame is not None and self._accept_frame(frame):
                    return
        except BaseException:
            try:
                self.disconnect()
            except BaseException:
                logger.warning("Subscriber cleanup failed while preserving connect failure", exc_info=True)
            raise

    def get_action(self) -> RobotAction:
        receiver = self._receiver
        if receiver is None:
            raise DeviceNotConnectedError(f"{self} is not connected")
        try:
            frame = receiver.receive_latest(timeout_s=0.0)
        except ValueError:
            self._invalidate_cache()
        else:
            if frame is not None:
                self._accept_frame(frame)
        if self._last_received_at is None:
            return neutral_action()
        if self._clock() - self._last_received_at > self.config.stale_after_s:
            self._armed = False
            return neutral_action()
        return normalize_action(self._latest_action)

    def disconnect(self) -> None:
        receiver, self._receiver = self._receiver, None
        try:
            if receiver is not None:
                receiver.close()
        finally:
            self._invalidate_cache()
            self._session_id = None
            self._last_sequence = -1
            self._armed = False

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def send_feedback(self, _feedback: dict[str, Any]) -> None:
        """The network subscriber has no haptic feedback channel."""

    def _accept_frame(self, frame: TeleopFrame) -> bool:
        if frame.session_id != self._session_id:
            self._session_id = frame.session_id
            self._last_sequence = -1
            self._armed = False
        if frame.sequence <= self._last_sequence:
            self._invalidate_cache()
            self._last_sequence = frame.sequence
            return False

        action = normalize_action(frame.action)
        self._last_sequence = frame.sequence
        self._latest_frame = frame
        self._last_received_at = self._clock()
        if not self._armed:
            self._armed = all(
                float(action[f"{side}.squeeze"]) <= self.config.rearm_squeeze_threshold
                for side in ("left", "right")
            )
        self._latest_action = action if self._armed else neutral_action()
        return True

    def _invalidate_cache(self) -> None:
        self._latest_frame = None
        self._latest_action = neutral_action()
        self._last_received_at = None
        self._armed = False


IsaacTeleopSubscriber = IsaacTeleopNodeSubscriber

__all__ = [
    "IsaacTeleopNodeConfig",
    "IsaacTeleopNodeSubscriber",
    "IsaacTeleopSubscriber",
]
