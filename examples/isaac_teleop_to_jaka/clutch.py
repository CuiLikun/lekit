#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Engage-relative clutch for the XR -> JAKA teleop loop.

Latches the controller pose and the EE home on the engage edge, then on every engaged
frame returns ``home + (grip - origin)`` for both position and orientation. Disengaging
holds the last computed EE pose so the arm does not creep; re-engaging captures a fresh
home from the measured pose so the arm does not snap back to a stale target.
"""

from __future__ import annotations

import numpy as np

from lerobot.utils.rotation import Rotation


class Clutch:
    """Per-engage origin latch for both position and orientation."""

    def __init__(self, home_base_T_ee: np.ndarray):  # noqa: N803
        home = np.asarray(home_base_T_ee, dtype=float)
        self._home_pos = home[:3, 3].copy()
        self._home_rot = Rotation.from_matrix(home[:3, :3])
        self._origin_pos = np.zeros(3, dtype=float)
        self._origin_rot = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))

    def engage(
        self,
        grip_pos: np.ndarray,
        grip_quat: np.ndarray,
        home_base_T_ee: np.ndarray | None = None,  # noqa: N803
    ) -> None:
        """Latch the controller origin; optionally update the EE home from a measured pose.

        Pass ``home_base_T_ee`` on each engage edge so the EE home tracks the measured pose
        (the arm may have moved while disengaged — e.g. gravity sag, external contact —
        and latching the last commanded pose would snap it back at servo speed).
        """
        if home_base_T_ee is not None:
            home = np.asarray(home_base_T_ee, dtype=float)
            self._home_pos = home[:3, 3].copy()
            self._home_rot = Rotation.from_matrix(home[:3, :3])
        self._origin_pos = np.asarray(grip_pos, dtype=float).copy()
        self._origin_rot = Rotation.from_quat(np.asarray(grip_quat, dtype=float))

    def rebase(
        self, grip_pos: np.ndarray, grip_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the absolute base-frame EE target ``(pos [m], quat [xyzw])`` for this frame."""
        pos = self._home_pos + (np.asarray(grip_pos, dtype=float) - self._origin_pos)
        rot_ctrl = Rotation.from_quat(np.asarray(grip_quat, dtype=float))
        rot = (rot_ctrl * self._origin_rot.inv()) * self._home_rot
        return pos, rot.as_quat()
