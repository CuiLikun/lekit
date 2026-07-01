#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Gamepad teleoperation of the SO101 arm in a MuJoCo simulation.

Uses the same IK-based Cartesian control pipeline as the real-robot teleop:
  left stick  -> XY end-effector motion
  right stick -> Z end-effector motion
  LB / RB     -> gripper close / open

No physical robot hardware is required -- only a gamepad and a display.
"""

import time
from pathlib import Path

import mujoco
import numpy as np

from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig

SO101_DIR = Path(__file__).resolve().parent.parent / "examples" / "phone_to_so100" / "SO101"

# ── constants ────────────────────────────────────────────────────────────────
FPS = 30

EE_BODY = "moving_jaw_so101_v1"
EE_STEP_SCALE = 0.005  # metres per gamepad unit per tick
IK_DAMPING = 0.02  # damped-least-squares lambda

JOINT_LO = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
JOINT_HI = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

GRIPPER_OPEN = 1.2  # radians
GRIPPER_CLOSE = 0.0  # radians


# ── IK (MuJoCo Jacobian) ────────────────────────────────────────────────────


def solve_ik_step(m, d, delta, joint_ids, body_id, damping):
    """Single damped-least-squares IK step. Returns delta_q (radians)."""
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, None, body_id)
    arm_jac = np.array([jacp[:, m.jnt_dofadr[jid]] for jid in joint_ids]).T
    jjt = arm_jac @ arm_jac.T + damping * np.eye(3)
    return arm_jac.T @ np.linalg.solve(jjt, delta)


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    import sys
    sys.path.insert(0, str(SO101_DIR))
    from interactive_viewer import SO101Viewer  # noqa: E402

    view = SO101Viewer(kp=2000.0, kv=3.0)
    m, d = view.model, view.data

    ee_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    joint_ids = list(range(5))

    # -- gamepad --
    teleop = GamepadTeleop(GamepadTeleopConfig(use_gripper=True))
    teleop.connect()
    if not teleop.is_connected:
        raise RuntimeError("Failed to connect gamepad")

    # -- state --
    target_q = np.array([d.qpos[m.jnt_qposadr[jid]] for jid in joint_ids])

    print("=" * 55)
    print("  SO101 Sim Teleop — Gamepad → MuJoCo IK")
    print("=" * 55)
    print("  Left stick   : XY end-effector motion")
    print("  Right stick  : Z end-effector motion")
    print("  LB           : gripper close")
    print("  RB           : gripper open")
    print("  Backspace    : reset pose")
    print("  Esc / q      : quit")
    print("=" * 55)

    while view.is_running:
        t0 = time.perf_counter()

        # ---- read gamepad ----
        action = teleop.get_action()
        dx = float(action.get("delta_x", 0.0))
        dy = float(action.get("delta_y", 0.0))
        dz = float(action.get("delta_z", 0.0))
        gripper_cmd = int(action.get("gripper", 1))

        # ---- gripper (instant) ----
        if gripper_cmd == 0:
            d.ctrl[5] = GRIPPER_CLOSE
        elif gripper_cmd == 2:
            d.ctrl[5] = GRIPPER_OPEN

        # ---- IK for arm joints ----
        norm = (dx**2 + dy**2 + dz**2) ** 0.5
        if norm > 1e-3:
            delta = np.array([dx, dy, dz]) * EE_STEP_SCALE
            dq = solve_ik_step(m, d, delta, joint_ids, ee_body_id, IK_DAMPING)
            target_q = np.clip(target_q + dq, JOINT_LO[:5], JOINT_HI[:5])

        view.send_action(target_q)

        # ---- step simulation ----
        for _ in range(4):
            mujoco.mj_step(m, d)
        view.step()

        # ---- pace ----
        elapsed = time.perf_counter() - t0
        remaining = 1.0 / FPS - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
