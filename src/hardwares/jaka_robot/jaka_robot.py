"""LeRobot-compatible driver for a 6-DOF JAKA robotic arm.

This module wraps the vendored :mod:`jkrc` Python SDK (``libjakaAPI.so`` and
``jkrc.so`` shipped in this directory). On Linux these shared libraries are
loaded via the ``LD_LIBRARY_PATH`` set up at import time below — keep them
next to this file when deploying.

Example:

    >>> from hardwares.jaka_robot import JakaRobot, JakaRobotConfig
    >>> cfg = JakaRobotConfig(ip="192.168.2.64")
    >>> robot = JakaRobot(cfg)
    >>> with robot:
    ...     obs = robot.get_observation()
    ...     robot.send_action({"joint_1.pos": 0.0, "joint_2.pos": 0.5, ...})

See https://www.jaka.com/docs/guide/1.7.2/SDK/Python.html for the upstream
SDK reference (1.7.2 controller family).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

# Add the directory containing the shared library to the LD_LIBRARY_PATH
# environment variable so ``libjakaAPI.so`` and ``jkrc.so`` can be located by
# the dynamic linker at module-import time. The dynamic linker only consults
# ``LD_LIBRARY_PATH`` at process start, so this only helps if the missing
# libraries are siblings of the .so on ``sys.path``; in most deployments the
# launcher script already sets the variable and this is a no-op.
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["LD_LIBRARY_PATH"] = f"{_script_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
# Add the directory to ``sys.path`` so the ``jkrc`` Python wrapper package
# can be located by the import system.
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import jkrc  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


# ── SDK Constants ────────────────────────────────────────────────────────────
# The JAKA Python SDK does not export any symbolic constants — every example
# in the vendor docs literalises them. We re-export them here so callers can
# import them by name.

# Move modes used by joint_move / linear_move / circular_move / servo_*.
ABS, INCR, CONT = 0, 1, 2

# Coordinate frames accepted by ``jog`` and friends.
COORD_BASE, COORD_JOINT, COORD_TOOL = 0, 1, 2

# Cartesian jog axes (matches ``aj_num`` 0..5 in cartesian coord types).
CART_X, CART_Y, CART_Z, CART_RX, CART_RY, CART_RZ = 0, 1, 2, 3, 4, 5

# IO group selectors (``iotype`` parameter).
IO_CABINET, IO_TOOL, IO_EXTEND = 0, 1, 2

# Program states returned by ``get_program_state``.
PROG_STOPPED, PROG_RUNNING, PROG_PAUSED = 0, 1, 2

# Motion planner types for ``set_motion_planner``.
PLANNER_DISABLED, PLANNER_T, PLANNER_S = -1, 0, 1

# Selective subset of controller error codes — surfaces a helpful message
# instead of an opaque ``ret[0]`` value. See the JAKA SDK Python docs for the
# complete table.
JAKA_ERROR_MESSAGES: dict[int, str] = {
    -1: "operation failed",
    -2: "invalid argument",
    -3: "communication failure or not connected",
    -4: "kinematic / inverse-kinematics failure",
    -5: "obstacle / collision detected",
    -6: "robot not powered on",
    -7: "robot not enabled",
    -8: "servo mode not enabled",
}


# ── Exception ────────────────────────────────────────────────────────────────


class JakaError(RuntimeError):
    """Raised when a JAKA SDK call returns a non-zero error code.

    Attributes:
        code: The integer error code returned by the SDK (i.e. ``ret[0]``).
        payload: The remainder of the SDK return tuple, when available.
    """

    def __init__(self, code: int, message: str = "", payload: tuple | None = None):
        self.code = int(code)
        self.payload = payload
        msg = message or JAKA_ERROR_MESSAGES.get(self.code, "unknown error")
        super().__init__(f"JAKA SDK error [{self.code}]: {msg}")


# ── Config ───────────────────────────────────────────────────────────────────


@RobotConfig.register_subclass("jaka_robot")
@dataclass
class JakaRobotConfig(RobotConfig):
    """Configuration for :class:`JakaRobot`.

    All fields are keyword-only (inherited from ``RobotConfig``). When loaded
    from YAML, ``type: jaka_robot`` selects this configuration.
    """

    # Controller IP address.
    ip: str = "192.168.1.31"

    # Default joint-space motion (rad/s, rad/s²). Used when ``move_j`` callers
    # do not override ``speed`` / ``acc`` explicitly.
    default_speed_joint: float = 0.5
    default_acc_joint: float = 3.5

    # Default Cartesian motion (mm/s, mm/s²).
    default_speed_linear: float = 100.0
    default_acc_linear: float = 500.0

    # Safety clamp applied to every ``send_action``: any joint whose target
    # differs from its current position by more than this magnitude is capped.
    # ``None`` disables clamping.
    max_relative_target: float | dict[str, float] | None = 0.05

    # Whether ``connect()`` should also power-on and enable the servos so the
    # first ``send_action`` is immediately ready.
    auto_power_on: bool = True
    auto_enable: bool = True

    # Use the gRPC channel for ``login`` when supported by the controller.
    use_grpc: bool = False

    # Default tool / user-frame IDs applied at connect time.
    tool_id: int = 0
    user_frame_id: int = 0

    # Collision protection level (0=disabled, 1..5 ⇒ 25/50/75/100/125 N).
    collision_level: int = 3

    # ``set_block_wait_timeout`` value used for blocking ``*_move`` calls.
    block_wait_timeout_s: float = 10.0

    def __post_init__(self):
        # Parent ensures cameras (if any) declare width/height/fps. JAKA ships
        # with no cameras, so the loop is a no-op.
        super().__post_init__()
        if not self.ip:
            raise ValueError("JakaRobotConfig.ip must not be empty.")
        if self.default_speed_joint <= 0:
            raise ValueError("JakaRobotConfig.default_speed_joint must be positive.")
        if self.default_acc_joint <= 0:
            raise ValueError("JakaRobotConfig.default_acc_joint must be positive.")
        if self.default_speed_linear <= 0:
            raise ValueError("JakaRobotConfig.default_speed_linear must be positive.")
        if self.max_relative_target is not None and not (
            (isinstance(self.max_relative_target, (int, float)) and self.max_relative_target > 0)
            or isinstance(self.max_relative_target, dict)
        ):
            raise ValueError(
                "JakaRobotConfig.max_relative_target must be a positive scalar or a "
                "dict of motor_name -> positive scalar, or None to disable."
            )
        if not 0 <= self.collision_level <= 5:
            raise ValueError("JakaRobotConfig.collision_level must be in [0, 5].")
        if self.tool_id < 0 or self.tool_id > 15:
            raise ValueError("JakaRobotConfig.tool_id must be in [0, 15].")
        if self.user_frame_id < 0 or self.user_frame_id > 15:
            raise ValueError("JakaRobotConfig.user_frame_id must be in [0, 15].")
        if self.block_wait_timeout_s < 0.5:
            raise ValueError("JakaRobotConfig.block_wait_timeout_s must be >= 0.5.")


# ── Robot implementation ─────────────────────────────────────────────────────


class JakaRobot(Robot):
    """LeRobot-compatible JAKA 6-DOF robotic arm driver.

    The class follows the standard :class:`lerobot.robots.robot.Robot`
    contract. ``get_observation`` returns the per-joint planned/actual
    positions, the TCP pose, and a handful of status flags. ``send_action``
    accepts a dict of per-joint target positions (radians), safety-clamps
    them against the current state, and dispatches a non-blocking
    incremental ``joint_move`` so successive calls can run at the policy
    control rate without stalling.

    For low-level access (drag mode, drag-mode IO, sinusoidal jog, servo
    streams, …) the underlying :class:`jkrc.RC` methods are exposed as
    thin wrappers — see the ``move_*``, ``servo_*``, ``set_drag_mode`` and
    IO helpers further down.
    """

    config_class: ClassVar[type] = JakaRobotConfig
    name: ClassVar[str] = "jaka_robot"

    # JAKA arms are 6-DOF. Joints are exposed under stable names so the
    # LeRobot observation/action schemas line up regardless of firmware.
    N_JOINTS: ClassVar[int] = 6
    motors: ClassVar[list[str]] = [f"joint_{i + 1}" for i in range(N_JOINTS)]
    TCP_AXES: ClassVar[tuple[str, ...]] = ("x", "y", "z", "rx", "ry", "rz")

    # ── Construction ──

    def __init__(self, config: JakaRobotConfig):
        super().__init__(config)
        self.config: JakaRobotConfig = config
        # Underlying SDK controller handle; populated in ``connect``.
        self.rc: Any | None = None

    # ── Feature schemas ──

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """Schema for everything ``get_observation`` emits."""
        features: dict[str, Any] = {f"{m}.pos": float for m in self.motors}
        features.update({f"{m}.actual_pos": float for m in self.motors})
        for axis in self.TCP_AXES:
            features[f"tcp.{axis}"] = float
            features[f"tcp.actual_{axis}"] = float
        features.update(
            {
                "status.powered_on": float,
                "status.enabled": float,
                "status.estop": float,
                "status.collision": float,
                "status.on_limit": float,
                "status.in_pos": float,
                "status.drag_mode": float,
                "status.rapidrate": float,
            }
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """Schema for the action dict ``send_action`` expects."""
        return {f"{m}.pos": float for m in self.motors}

    # ── Connection state ──

    @property
    def is_connected(self) -> bool:
        """True once ``connect`` has established a live RC handle."""
        return self.rc is not None

    @property
    def is_calibrated(self) -> bool:
        # The controller handles joint calibration internally; the host has
        # no per-motor calibration step.
        return True

    def calibrate(self) -> None:
        """No-op: JAKA controllers self-calibrate; nothing to do here."""
        return

    # ── Lifecycle ──

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the controller, log in, optionally power-on and enable.

        Args:
            calibrate: Accepted for interface parity — ignored. JAKA has no
                host-side calibration step to run.
        """
        del calibrate  # Unused — see docstring.

        rc = jkrc.RC(self.config.ip)
        # The vendor docs and the upstream ``login``/``logout`` names disagree
        # with the .pyi stubs (``log_in``/``log_out``). Probe at runtime and
        # fall back so we work with either binding.
        login = getattr(rc, "login", None) or rc.log_in
        use_grpc = 1 if self.config.use_grpc else 0
        try:
            self._check(login(use_grpc=use_grpc))
        except TypeError:
            # The legacy TCP-only binding takes no ``use_grpc`` argument.
            self._check(login())
        logger.info("JakaRobot[%s] logged in (use_grpc=%s)", self.config.ip, self.config.use_grpc)
        self.rc = rc

        if self.config.auto_power_on:
            self.power_on()
        if self.config.auto_enable:
            self.enable_robot()

        # Default IDs / safety settings.
        self._check(self.rc.set_tool_id(self.config.tool_id))
        self._check(self.rc.set_user_frame_id(self.config.user_frame_id))
        self._check(self.rc.set_collision_level(self.config.collision_level))
        # ``set_block_wait_timeout`` is deprecated but still respected by some
        # firmware versions; ignore errors to stay forward-compatible.
        try:
            self._check(self.rc.set_block_wait_timeout(self.config.block_wait_timeout_s))
        except JakaError:
            logger.debug("JakaRobot: set_block_wait_timeout unsupported, ignoring.")

        self.configure()
        logger.info("JakaRobot[%s] connected and ready.", self.config.ip)

    def configure(self) -> None:
        """Apply runtime configuration. Called once at the end of ``connect``.

        Override-friendly — subclasses may replace this to install custom
        planners, drag-mode friction coefficients, payload settings, etc.
        """
        if not self.is_connected:
            return
        # Default to the T-planner (speed priority). Callers can flip this at
        # runtime via ``set_motion_planner``.
        self._check(self.rc.set_motion_planner(PLANNER_T))

    @check_if_not_connected
    def disconnect(self) -> None:
        """Best-effort teardown: disable, log out, drop the handle."""
        rc = self.rc
        try:
            try:
                # ``payload = (ctrl_errcode, errmsg, powered_on, enabled)``
                payload = self._check(rc.get_robot_status_simple(), return_payload=True)
                if bool(payload[3]):
                    self._check(rc.disable_robot())
            except JakaError:
                pass
            logout = getattr(rc, "logout", None) or rc.log_out
            self._check(logout())
        except JakaError as e:
            logger.warning("JakaRobot: error during disconnect: %s", e)
        finally:
            self.rc = None
            logger.info("JakaRobot[%s] disconnected.", self.config.ip)

    # ── Return-tuple validation ──

    def _check(self, ret, return_payload: bool = False):
        """Validate the error code at ``ret[0]``; optionally yield the payload.

        All JAKA SDK calls return a tuple whose first element is an integer
        error code (0 = success). On any non-zero value this raises
        :class:`JakaError`. With ``return_payload=True`` it instead returns
        the data portion of the return value as a flat tuple.

        The SDK uses two success shapes:

            * nested: ``(0, payload_tuple)`` (e.g. ``get_joint_position``)
            * flat:   ``(0, val1, val2, ...)`` (e.g. ``get_rapidrate``)

        For the nested case the inner tuple is unwrapped once, so callers
        always get a flat sequence they can index directly.
        """
        if ret is None:
            raise JakaError(-1, "SDK call returned None")
        # ``jkrc`` returns plain tuples; be defensive about types anyway.
        try:
            code = int(ret[0])
        except (TypeError, IndexError, ValueError) as e:
            raise JakaError(-1, f"unparseable SDK return: {ret!r}") from e
        tail = tuple(ret[1:]) if len(ret) > 1 else ()
        # Unwrap a single nested tuple: ``(0, (a, b, c))`` → ``(a, b, c)``.
        if len(tail) == 1 and isinstance(tail[0], (tuple, list)):
            tail = tuple(tail[0])
        if code != 0:
            raise JakaError(code, payload=tail)
        if return_payload:
            return tail
        return None

    # ── Observation / Action ──

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read the current joint/TCP state and a few status flags."""
        rc = self.rc

        # Joint positions (planned + servo-feedback).
        jp = self._check(rc.get_joint_position(), return_payload=True)
        jp_actual = self._check(rc.get_actual_joint_position(), return_payload=True)
        # TCP pose (planned + servo-feedback).
        tcp = self._check(rc.get_tcp_position(), return_payload=True)
        tcp_actual = self._check(rc.get_actual_tcp_position(), return_payload=True)

        # Status flags — best-effort: a flag we can't read is simply set to 0
        # rather than failing the whole observation. The core joint/TCP reads
        # above still raise on real errors.
        powered_on = enabled = 0.0
        estop = collision = on_limit = in_pos = drag = 0.0
        rapidrate = 1.0
        try:
            payload = self._check(rc.get_robot_status_simple(), return_payload=True)
            # payload = (ctrl_errcode, errmsg, powered_on, enabled)
            powered_on = float(payload[2])
            enabled = float(payload[3])
        except JakaError:
            pass
        for src, sink in (
            ((rc.is_in_estop,), "estop"),
            ((rc.is_in_collision,), "collision"),
            ((rc.is_on_limit,), "on_limit"),
            ((rc.is_in_pos,), "in_pos"),
            ((rc.is_in_drag_mode,), "drag"),
        ):
            try:
                payload = self._check(src[0](), return_payload=True)
                val = float(payload[0])
            except JakaError:
                val = 0.0
            if sink == "estop":
                estop = val
            elif sink == "collision":
                collision = val
            elif sink == "on_limit":
                on_limit = val
            elif sink == "in_pos":
                in_pos = val
            elif sink == "drag":
                drag = val
        try:
            payload = self._check(rc.get_rapidrate(), return_payload=True)
            rapidrate = float(payload[0])
        except JakaError:
            pass

        obs: dict[str, Any] = {}
        for i, m in enumerate(self.motors):
            obs[f"{m}.pos"] = float(jp[i])
            obs[f"{m}.actual_pos"] = float(jp_actual[i])
        for i, axis in enumerate(self.TCP_AXES):
            obs[f"tcp.{axis}"] = float(tcp[i])
            obs[f"tcp.actual_{axis}"] = float(tcp_actual[i])
        obs["status.powered_on"] = powered_on
        obs["status.enabled"] = enabled
        obs["status.estop"] = estop
        obs["status.collision"] = collision
        obs["status.on_limit"] = on_limit
        obs["status.in_pos"] = in_pos
        obs["status.drag_mode"] = drag
        obs["status.rapidrate"] = rapidrate
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Drive each named joint toward ``action[m]`` via incremental joint moves.

        Steps:
            1. Snapshot the actual joint positions from the controller.
            2. For every key present in ``action``, clip the goal against
               ``config.max_relative_target`` (per joint, rad). Joints absent
               from ``action`` are left untouched (zero delta).
            3. Issue a non-blocking ``joint_move(INCR)`` of the clipped
               delta at ``config.default_speed_joint``.

        Returns the (clamped) action that was actually sent to the controller
        so the caller can log or compare it against the raw policy output.
        """
        rc = self.rc

        actual = self._check(rc.get_actual_joint_position(), return_payload=True)
        present = {f"{m}.pos": float(actual[i]) for i, m in enumerate(self.motors)}

        # Filter down to the joints the caller is actually driving. Unknown
        # keys are ignored so the caller can pass arbitrary telemetry fields.
        goals: dict[str, float] = {}
        for i, m in enumerate(self.motors):
            key = f"{m}.pos"
            if key in action:
                try:
                    goals[key] = float(action[key])
                except (TypeError, ValueError) as e:
                    raise TypeError(f"JakaRobot.send_action: action[{key!r}]={action[key]!r} is not numeric") from e

        if goals:
            safe = ensure_safe_goal_position(
                {k: (goals[k], present[k]) for k in goals},
                self.config.max_relative_target,
            )
        else:
            safe = {}

        # Build a 6-tuple delta in joint order. Joints the caller didn't touch
        # (or that aren't even in the action dict) get a zero delta so the
        # motion queue never sees ``None`` entries.
        cmd = tuple(
            safe.get(f"joint_{i + 1}.pos", present[f"joint_{i + 1}.pos"]) - present[f"joint_{i + 1}.pos"]
            for i in range(self.N_JOINTS)
        )
        self._check(rc.joint_move(cmd, INCR, False, self.config.default_speed_joint))

        # Echo back the values actually requested (post-clamp), filling any
        # joints the caller didn't include with their current position so the
        # result has the same shape as ``action_features``.
        return {**present, **safe}

    # ── Direct motion commands (bypass the per-step safety clamp) ──

    @check_if_not_connected
    def move_j(
        self,
        joint_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        is_block: bool = True,
        speed: float | None = None,
        acc: float | None = None,
        tol: float | None = None,
    ) -> None:
        """Joint-space ``joint_move``. ``joint_pos`` in radians."""
        if acc is None:
            self._check(
                self.rc.joint_move(
                    tuple(joint_pos),
                    move_mode,
                    is_block,
                    speed if speed is not None else self.config.default_speed_joint,
                )
            )
            return
        self._check(
            self.rc.joint_move_extend(
                tuple(joint_pos),
                move_mode,
                is_block,
                speed if speed is not None else self.config.default_speed_joint,
                acc,
                tol if tol is not None else 0.1,
            )
        )

    @check_if_not_connected
    def move_l(
        self,
        tcp_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        is_block: bool = True,
        speed: float | None = None,
        acc: float | None = None,
        tol: float | None = None,
        ori_vel: float | None = None,
        ori_acc: float | None = None,
    ) -> None:
        """Cartesian ``linear_move``. ``tcp_pos`` is ``[x, y, z, rx, ry, rz]``
        in mm / rad."""
        linear_speed = speed if speed is not None else self.config.default_speed_linear
        if acc is None:
            self._check(self.rc.linear_move(tuple(tcp_pos), move_mode, is_block, linear_speed))
            return
        # The ``linear_move_extend_ori`` variant lets us tune translational
        # and orientational accels independently; the SDK picks it if the
        # caller opts into orientation limits.
        if ori_vel is not None or ori_acc is not None:
            self._check(
                self.rc.linear_move_extend_ori(
                    tuple(tcp_pos),
                    move_mode,
                    is_block,
                    linear_speed,
                    acc,
                    tol if tol is not None else 0.5,
                    ori_vel if ori_vel is not None else 3.141592653589793,
                    ori_acc if ori_acc is not None else 12.566370614359172,
                )
            )
            return
        self._check(
            self.rc.linear_move_extend(
                tuple(tcp_pos),
                move_mode,
                is_block,
                linear_speed,
                acc,
                tol if tol is not None else 0.5,
            )
        )

    @check_if_not_connected
    def move_c(
        self,
        end_pos: list[float] | tuple[float, ...],
        mid_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        is_block: bool = True,
        speed: float | None = None,
        acc: float | None = None,
        tol: float | None = None,
        circle_cnt: int = 1,
        circle_mode: int | None = None,
    ) -> None:
        """Circular ``circular_move`` via a mid-point pose (mm / rad)."""
        linear_speed = speed if speed is not None else self.config.default_speed_linear
        linear_acc = acc if acc is not None else 1.0
        linear_tol = tol if tol is not None else 0.5
        if circle_mode is None:
            self._check(
                self.rc.circular_move_extend(
                    tuple(end_pos),
                    tuple(mid_pos),
                    move_mode,
                    is_block,
                    linear_speed,
                    linear_acc,
                    linear_tol,
                    None,
                    int(circle_cnt),
                )
            )
            return
        self._check(
            self.rc.circular_move_extend_mode(
                tuple(end_pos),
                tuple(mid_pos),
                move_mode,
                is_block,
                linear_speed,
                linear_acc,
                linear_tol,
                None,
                int(circle_cnt),
                int(circle_mode),
            )
        )

    @check_if_not_connected
    def jog(
        self,
        aj_num: int,
        jog_vel: float,
        move_mode: int = CONT,
        coord_type: int = COORD_BASE,
        pos_cmd: float = 0.0,
    ) -> None:
        """Manual-mode ``jog``. With ``move_mode = CONT`` motion continues
        until :meth:`jog_stop` is called."""
        self._check(self.rc.jog(int(aj_num), int(move_mode), int(coord_type), float(jog_vel), float(pos_cmd)))

    @check_if_not_connected
    def jog_stop(self, joint_num: int = -1) -> None:
        """Halt one (``0..5``) or all (``-1``) jogging axes."""
        self._check(self.rc.jog_stop(int(joint_num)))

    @check_if_not_connected
    def motion_abort(self) -> None:
        """Force-stop the currently executing motion."""
        self._check(self.rc.motion_abort())

    @check_if_not_connected
    def is_in_pos(self) -> bool:
        """Return True if the arm has settled at its target."""
        return bool(self._check(self.rc.is_in_pos(), return_payload=True)[0])

    @check_if_not_connected
    def get_motion_status(self) -> dict[str, Any]:
        """Snapshot of the controller's motion-queue telemetry."""
        payload = self._check(self.rc.get_motion_status(), return_payload=True)
        keys = (
            "motion_line",
            "motion_line_sdk",
            "inpos",
            "err_add_line",
            "queue",
            "active_queue",
            "queue_full",
            "paused",
            "on_limit",
            "in_estop",
            "in_collision",
        )
        return {k: payload[i] for i, k in enumerate(keys) if i < len(payload)}

    # ── Power / enable / drag ──

    @check_if_not_connected
    def power_on(self) -> None:
        """Power on the arm (~8 s delay)."""
        self._check(self.rc.power_on())
        logger.info("JakaRobot: powered on")

    @check_if_not_connected
    def power_off(self) -> None:
        """Power off the arm."""
        self._check(self.rc.power_off())

    @check_if_not_connected
    def enable_robot(self) -> None:
        """Engage the servos."""
        self._check(self.rc.enable_robot())
        logger.info("JakaRobot: enabled")

    @check_if_not_connected
    def disable_robot(self) -> None:
        """Disengage the servos."""
        self._check(self.rc.disable_robot())

    @check_if_not_connected
    def shutdown(self) -> None:
        """Shut down the controller cabinet."""
        self._check(self.rc.shut_down())

    @check_if_not_connected
    def set_drag_mode(self, enable: bool) -> None:
        """Toggle gravity-compensated hand-guidance (drag) mode."""
        self._check(self.rc.drag_mode_enable(bool(enable)))

    @check_if_not_connected
    def is_in_drag_mode(self) -> bool:
        return bool(self._check(self.rc.is_in_drag_mode(), return_payload=True)[0])

    @check_if_not_connected
    def set_rapidrate(self, rate: float) -> None:
        """Set the global speed multiplier in ``[0.0, 1.0]``."""
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"rapidrate must be in [0, 1], got {rate}")
        self._check(self.rc.set_rapidrate(float(rate)))

    @check_if_not_connected
    def get_rapidrate(self) -> float:
        return float(self._check(self.rc.get_rapidrate(), return_payload=True)[0])

    @check_if_not_connected
    def set_motion_planner(self, planner_type: int = PLANNER_T) -> None:
        """Switch motion planner: ``PLANNER_DISABLED`` (``-1``), ``PLANNER_T``
        (``0``), ``PLANNER_S`` (``1``)."""
        if planner_type not in (PLANNER_DISABLED, PLANNER_T, PLANNER_S):
            raise ValueError(f"planner_type must be one of {PLANNER_DISABLED, PLANNER_T, PLANNER_S}")
        self._check(self.rc.set_motion_planner(int(planner_type)))

    # ── Tool / user-frame ──

    @check_if_not_connected
    def set_tool_id(self, id: int) -> None:
        if not 0 <= int(id) <= 15:
            raise ValueError("tool id must be in [0, 15]")
        self._check(self.rc.set_tool_id(int(id)))

    @check_if_not_connected
    def get_tool_id(self) -> int:
        return int(self._check(self.rc.get_tool_id(), return_payload=True)[0])

    @check_if_not_connected
    def set_user_frame_id(self, id: int) -> None:
        if not 0 <= int(id) <= 15:
            raise ValueError("user frame id must be in [0, 15]")
        self._check(self.rc.set_user_frame_id(int(id)))

    @check_if_not_connected
    def get_user_frame_id(self) -> int:
        return int(self._check(self.rc.get_user_frame_id(), return_payload=True)[0])

    @check_if_not_connected
    def set_tool_data(self, id: int, tcp: list[float], name: str = "") -> None:
        """Define a tool's TCP offset ``[x, y, z, rx, ry, rz]`` (mm / rad)."""
        self._check(self.rc.set_tool_data(int(id), tuple(tcp), name))

    @check_if_not_connected
    def get_tool_data(self, id: int) -> tuple[int, tuple[float, ...]]:
        payload = self._check(self.rc.get_tool_data(int(id)), return_payload=True)
        return int(payload[0]), tuple(payload[1])

    @check_if_not_connected
    def set_user_frame_data(self, id: int, frame: list[float], name: str = "") -> None:
        """Define a user-frame origin ``[x, y, z, rx, ry, rz]`` (mm / rad)."""
        self._check(self.rc.set_user_frame_data(int(id), tuple(frame), name))

    @check_if_not_connected
    def get_user_frame_data(self, id: int) -> tuple[int, tuple[float, ...]]:
        payload = self._check(self.rc.get_user_frame_data(int(id)), return_payload=True)
        return int(payload[0]), tuple(payload[1])

    # ── Safety / error utilities ──

    @check_if_not_connected
    def is_estopped(self) -> bool:
        return bool(self._check(self.rc.is_in_estop(), return_payload=True)[0])

    @check_if_not_connected
    def is_on_limit(self) -> bool:
        return bool(self._check(self.rc.is_on_limit(), return_payload=True)[0])

    @check_if_not_connected
    def is_in_collision(self) -> bool:
        return bool(self._check(self.rc.is_in_collision(), return_payload=True)[0])

    @check_if_not_connected
    def recover_collision(self) -> None:
        """Exit collision-protection mode after clearing the cause."""
        self._check(self.rc.collision_recover())

    @check_if_not_connected
    def clear_error(self) -> None:
        """Reset controller-side error state."""
        self._check(self.rc.clear_error())

    @check_if_not_connected
    def set_collision_level(self, level: int) -> None:
        if not 0 <= int(level) <= 5:
            raise ValueError("collision level must be in [0, 5]")
        self._check(self.rc.set_collision_level(int(level)))

    @check_if_not_connected
    def get_collision_level(self) -> int:
        return int(self._check(self.rc.get_collision_level(), return_payload=True)[0])

    @check_if_not_connected
    def get_last_error(self) -> int:
        return int(self._check(self.rc.get_last_error(), return_payload=True)[0])

    # ── Servo-mode (fast closed-loop) commands ──

    @check_if_not_connected
    def servo_enable(self, enable: bool = True) -> None:
        """Toggle servo mode. Required before :meth:`servo_j` / :meth:`servo_p`."""
        self._check(self.rc.servo_move_enable(bool(enable), True))

    @check_if_not_connected
    def is_in_servo(self) -> bool:
        return bool(self._check(self.rc.is_in_servomove(), return_payload=True)[0])

    @check_if_not_connected
    def servo_j(
        self,
        joint_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        step_num: int = 1,
    ) -> None:
        """Push one joint-space servo target. Period = ``step_num * 8 ms``."""
        self._check(self.rc.servo_j(tuple(joint_pos), int(move_mode), int(step_num)))

    @check_if_not_connected
    def servo_p(
        self,
        tcp_pos: list[float] | tuple[float, ...],
        move_mode: int = ABS,
        step_num: int = 1,
    ) -> None:
        """Push one Cartesian servo target (mm / rad)."""
        self._check(self.rc.servo_p(tuple(tcp_pos), int(move_mode), int(step_num)))

    # ── IO helpers (work on cabinet / tool / extended IO blocks) ──

    @staticmethod
    def _io_kind() -> dict[str, int]:
        return {"cabinet": IO_CABINET, "tool": IO_TOOL, "extend": IO_EXTEND}

    @check_if_not_connected
    def get_digital_input(self, iotype: int, index: int) -> int:
        return int(self._check(self.rc.get_digital_input(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def get_digital_output(self, iotype: int, index: int) -> int:
        return int(self._check(self.rc.get_digital_output(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def set_digital_output(self, iotype: int, index: int, value: bool | int) -> None:
        self._check(self.rc.set_digital_output(int(iotype), int(index), bool(value)))

    @check_if_not_connected
    def get_analog_input(self, iotype: int, index: int) -> float:
        return float(self._check(self.rc.get_analog_input(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def get_analog_output(self, iotype: int, index: int) -> float:
        return float(self._check(self.rc.get_analog_output(int(iotype), int(index)), return_payload=True)[0])

    @check_if_not_connected
    def set_analog_output(self, iotype: int, index: int, value: float) -> None:
        self._check(self.rc.set_analog_output(int(iotype), int(index), float(value)))

    # ── App-script (JAKA ``*.jks`` program) interface ──

    @check_if_not_connected
    def program_load(self, name: str) -> None:
        self._check(self.rc.program_load(name))

    @check_if_not_connected
    def program_run(self) -> None:
        self._check(self.rc.program_run())

    @check_if_not_connected
    def program_pause(self) -> None:
        self._check(self.rc.program_pause())

    @check_if_not_connected
    def program_resume(self) -> None:
        self._check(self.rc.program_resume())

    @check_if_not_connected
    def program_abort(self) -> None:
        self._check(self.rc.program_abort())

    @check_if_not_connected
    def get_loaded_program(self) -> str:
        return str(self._check(self.rc.get_loaded_program(), return_payload=True)[0])

    @check_if_not_connected
    def get_program_state(self) -> int:
        """Return one of ``PROG_STOPPED`` (0), ``PROG_RUNNING`` (1), ``PROG_PAUSED`` (2)."""
        return int(self._check(self.rc.get_program_state(), return_payload=True)[0])

    # ── Misc ──

    @check_if_not_connected
    def get_sdk_version(self) -> str:
        """Return the SDK version string (e.g. ``"1.7.2"``)."""
        payload = self._check(self.rc.get_sdk_version(), return_payload=True)
        version = payload[0]
        if isinstance(version, (tuple, list)):
            return ".".join(str(x) for x in version)
        return str(version)

    @check_if_not_connected
    def get_controller_ip(self) -> list[str]:
        """Return controller IPs reachable from this client."""
        payload = self._check(self.rc.get_controller_ip(), return_payload=True)
        ip_list = payload[0]
        return list(ip_list) if isinstance(ip_list, (tuple, list)) else [str(ip_list)]

    @property
    def controller_ip(self) -> str:
        return self.config.ip

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return f"JakaRobot(ip={self.config.ip!r}, {state})"


__all__ = [
    "ABS",
    "CART_RX",
    "CART_RY",
    "CART_RZ",
    "CART_X",
    "CART_Y",
    "CART_Z",
    "CONT",
    "COORD_BASE",
    "COORD_JOINT",
    "COORD_TOOL",
    "INCR",
    "IO_CABINET",
    "IO_EXTEND",
    "IO_TOOL",
    "JAKA_ERROR_MESSAGES",
    "PLANNER_DISABLED",
    "PLANNER_S",
    "PLANNER_T",
    "PROG_PAUSED",
    "PROG_RUNNING",
    "PROG_STOPPED",
    "JakaError",
    "JakaRobot",
    "JakaRobotConfig",
]


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    cfg = JakaRobotConfig(ip="192.168.1.31")
    robot = JakaRobot(cfg)
    print(robot)
    with robot:
        print("SDK:", robot.get_sdk_version())
        for _ in range(5):
            obs = robot.get_observation()
            jp = np.array([obs[f"joint_{i + 1}.actual_pos"] for i in range(6)])
            print(f"joints: {np.round(jp, 3)}")
            cmd = {f"joint_{i + 1}.pos": jp[i] + 0.01 for i in range(6)}
            robot.send_action(cmd)
            time.sleep(0.1)
