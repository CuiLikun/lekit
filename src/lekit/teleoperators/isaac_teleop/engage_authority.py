"""Edge-triggered Hub authority for one Quest controller hand."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lekit.control.controller import ControllerNode, HandleExpired, HandleNotGranted
from lekit.control.model import ControlHandle

from .protocol import CONTROLLER_SIDES


class EngageAuthority:
    """Turn one hand's Engage state into safe Controller authority requests.

    A grant is deliberately inert until a *tracked release* has been observed.
    This prevents a squeeze already held during a Hub reconnect from immediately
    taking over a robot.  Tracking loss is treated as a release of authority and
    requires the same release-then-press sequence when tracking returns.
    """

    def __init__(self, hand: str = "right") -> None:
        if hand not in CONTROLLER_SIDES:
            raise ValueError(f"hand must be one of {CONTROLLER_SIDES!r}")
        self.hand = hand
        self._handle_id: str | None = None
        self._handle: ControlHandle | None = None
        self._armed_after_release = False
        self._previous_engaged = False
        self._controlling = False

    def update(self, action: Mapping[str, Any], controller: ControllerNode) -> None:
        """Apply one normalized XR action before its managed frame is published."""

        handle = controller.current_handle
        if handle is None:
            self._release(controller)
            self._clear()
            return

        if handle.handle_id != self._handle_id:
            self._release(controller)
            self._handle_id = handle.handle_id
            self._handle = handle
            self._armed_after_release = False
            self._previous_engaged = False
            self._controlling = False

        tracking = bool(action[f"{self.hand}.is_tracking"])
        engaged = bool(action[f"{self.hand}.is_engaged"])
        if not tracking:
            self._release(controller)
            self._armed_after_release = False
            self._previous_engaged = False
            return

        if not engaged:
            self._release(controller)
            self._armed_after_release = True
        elif self._armed_after_release and not self._previous_engaged and not self._controlling:
            try:
                controller.take_over(handle)
            except (HandleNotGranted, HandleExpired):
                # The Hub can revoke a Handle concurrently with input sampling.
                # Treat it as a harmless failed edge; a later release rearms us.
                self._controlling = False
            else:
                self._controlling = True
        self._previous_engaged = engaged

    def reset(self, controller: ControllerNode, *, release: bool) -> None:
        """Forget local edge state, optionally releasing active authority once."""

        if release:
            self._release(controller)
        self._clear()

    def _release(self, controller: ControllerNode) -> None:
        if not self._controlling or self._handle is None:
            return
        try:
            controller.hand_over(self._handle)
        except (HandleNotGranted, HandleExpired):
            # A concurrent revoke has already completed this release logically.
            pass
        finally:
            self._controlling = False

    def _clear(self) -> None:
        self._handle_id = None
        self._handle = None
        self._armed_after_release = False
        self._previous_engaged = False
        self._controlling = False


__all__ = ["EngageAuthority"]
