"""SimRobot -- a MuJoCo-backed, LeRobot-compatible robot.

This module implements the LeRobot v0.6 ``Robot`` interface on top of an MJCF
model so a simulated arm can be used anywhere a physical robot would:
dataset recording (``lerobot.record``), evaluation (``lerobot.evaluate``),
policy rollout, teleoperation, etc.

It follows the *bring-your-own-hardware* integration guide
(https://huggingface.co/docs/lerobot/v0.6.0/en/integrate_hardware):

1. A ``RobotConfig`` subclass registered with draccus so ``type: sim_robot``
   resolves through the LeRobot factory.
2. A ``Robot`` subclass that implements the abstract methods.
3. Optional integration through ``lerobot.robots.utils.make_robot_from_config``
   once the module has been imported.

Quickstart
----------

.. code-block:: python

    from pathlib import Path
    from lerobot.robots.utils import make_robot_from_config

    from hardwares.sim_robot import (
        SimCameraConfig,
        SimMotorSpec,
        SimRobotConfig,
    )

    cfg = SimRobotConfig(
        id="so101_sim",
        xml_path=Path("/path/to/so101_scene.xml"),
        motors={
            "shoulder_pan":  SimMotorSpec(joint="joint1"),
            "shoulder_lift": SimMotorSpec(joint="joint2"),
            "elbow_flex":    SimMotorSpec(joint="joint3"),
            "wrist_flex":    SimMotorSpec(joint="joint4"),
            "wrist_roll":    SimMotorSpec(joint="joint5"),
            "gripper":       SimMotorSpec(joint="joint6"),
        },
        cameras={
            "top": SimCameraConfig(name="top_cam", width=640, height=480, fps=30),
            "wrist": SimCameraConfig(name="wrist_cam", width=480, height=640, fps=30),
        },
        default_joint_positions={
            "shoulder_pan":  0.0,
            "shoulder_lift": -1.57,
            "elbow_flex":    1.57,
            "wrist_flex":    0.0,
            "wrist_roll":    0.0,
            "gripper":       0.5,
        },
        max_relative_target=0.05,  # ~3 deg per step safety clamp
        n_substeps=4,
    )

    robot = make_robot_from_config(cfg)  # or: SimRobot(cfg)
    with robot:
        obs = robot.get_observation()
        # obs["shoulder_pan.pos"] -> float (radians)
        # obs["top"] -> np.ndarray[H, W, 3] uint8 RGB
        robot.send_action({
            "shoulder_pan.pos":  0.1,
            "shoulder_lift.pos": -1.4,
            ...
        })

Headless rendering
------------------

MuJoCo's offscreen renderer needs an OpenGL backend. Set
``MUJOCO_GL=egl`` (or ``osmesa``) before importing this module on a
headless Linux machine::

    export MUJOCO_GL=egl

On desktops the default ``glfw`` backend works as long as a display is
available.

Joint semantics
---------------

Each entry in ``motors`` maps a *motor name* (used in observation/action
dictionaries as ``"{motor}.pos"``) to a MuJoCo *joint*. If the model's MJCF
also defines an actuator for that joint, the actuator's id is resolved
automatically; otherwise the joint position is set kinematically via
``data.qpos``. Each motor is assumed to be a 1-DOF (hinge/slide) joint --
the value used for that joint is ``data.qpos[jnt_qposadr[jid]]``. Mobile-base
free joints are not currently supported.

Camera semantics
----------------

``cameras`` maps a *camera name* (used as the image key in
``get_observation``) to a MuJoCo ``<camera>`` element. Frames are rendered
offscreen through ``mujoco.Renderer`` at the requested ``width``/``height``
and returned as ``uint8`` ``(H, W, 3)`` RGB arrays.

Units
-----

Joint positions and actions are exchanged in radians by default (matching
MuJoCo's internal representation). Set ``use_degrees=True`` to convert at the
API boundary.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import (
    check_if_already_connected,
    check_if_not_connected,
)
# LeRobot backend visualization (rerun / foxglove) — lazy-imported so the
# optional ``viz`` extras don't become a hard requirement for SimRobot.
from lerobot.utils.visualization_utils import (
    VISUALIZATION_MODES,
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

logger = logging.getLogger(__name__)


# ── Lazy mujoco import ──────────────────────────────────────────────────────
# mujoco is only required when SimRobot is *used* (not when the module is
# merely imported). This keeps ``from hardwares.sim_robot import …`` working
# in environments where mujoco is not installed.

_MUJOCO = None


def _require_mujoco():
    """Import mujoco lazily and cache the module."""
    global _MUJOCO
    if _MUJOCO is None:
        try:
            import mujoco as _m
        except ImportError as e:
            raise ImportError(
                "mujoco is required to use SimRobot. Install it with "
                "`pip install mujoco` or `uv pip install mujoco`."
            ) from e
        _MUJOCO = _m
    return _MUJOCO


# ── Configs ─────────────────────────────────────────────────────────────────


@dataclass
class SimMotorSpec:
    """Specification for a single actuated joint.

    Attributes:
        joint: Name of the ``<joint>`` element in the MJCF.
        actuator: Optional name of the ``<actuator>`` element driving the
            joint. If ``None``, the simulator falls back to setting the joint
            position directly on ``data.qpos`` (kinematic, no dynamics).
        initial_position: Optional initial value for the joint. Overrides
            ``default_joint_positions`` for this single motor when supplied.
        ctrl_low: Optional lower bound applied to ``data.ctrl`` before stepping.
            If ``None``, the actuator's own ``ctrlrange`` is used.
        ctrl_high: Optional upper bound applied to ``data.ctrl`` before stepping.
            If ``None``, the actuator's own ``ctrlrange`` is used.
        kp: Optional proportional gain for the position-servo override. When
            ``enable_position_servos`` is ``True`` on the parent config, this
            ``kp`` (with ``kv``) is applied via the actuator's ``gainprm`` /
            ``biasprm`` slots, converting a generic ``<motor>`` actuator into a
            position servo at connect-time. ``None`` falls back to a sensible
            default.
        kv: Optional derivative gain for the position-servo override (see
            ``kp``).
    """

    joint: str
    actuator: str | None = None
    initial_position: float | None = None
    ctrl_low: float | None = None
    ctrl_high: float | None = None
    kp: float | None = None
    kv: float | None = None


@dataclass
class SimCameraConfig:
    """Specification for a simulated camera.

    Camera frames are rendered offscreen from MuJoCo's ``<camera>`` elements
    via ``mujoco.Renderer``.

    Attributes:
        name: Name of the ``<camera>`` element in the MJCF.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Informational; not enforced by SimRobot itself.
    """

    name: str
    width: int = 640
    height: int = 480
    fps: int = 30


@RobotConfig.register_subclass("sim_robot")
@dataclass
class SimRobotConfig(RobotConfig):
    """Configuration for :class:`SimRobot`.

    All fields are keyword-only (inherited from ``RobotConfig``). When loaded
    from YAML, ``type: sim_robot`` selects this configuration.
    """

    # Path to the MuJoCo scene (XML/MJCF).
    xml_path: str = ""
    # motor_name -> SimMotorSpec. The motor name appears as ``"{motor}.pos"``
    # in observation and action dictionaries.
    motors: dict[str, SimMotorSpec] = field(default_factory=dict)
    # camera_name -> SimCameraConfig. The camera name appears as an image key
    # in ``get_observation``. Frames are rendered offscreen from MJCF cameras.
    cameras: dict[str, SimCameraConfig] = field(default_factory=dict)
    # Default starting joint positions keyed by *motor* name. Applied on every
    # ``configure()`` / ``connect()`` call. Units depend on ``use_degrees``.
    default_joint_positions: dict[str, float] | None = None
    # Optional initial ``data.qpos`` (overrides ``default_joint_positions``
    # when set). Must follow MuJoCo's ``qpos`` layout for the model.
    init_qpos: list[float] | None = None
    # Number of ``mj_step`` calls per ``send_action`` invocation. Higher
    # values produce faster-than-real-time virtual dynamics.
    n_substeps: int = 1
    # When True, ``SimRobot.connect`` rewrites each motor's ``gainprm`` /
    # ``biasprm`` to behave as a critically-damped position servo. Useful for
    # generic MJCFs that ship plain ``<motor>`` actuators (e.g. the SO-101
    # assets in this repo). The ``SimMotorSpec.kp``/``kv`` per-motor values
    # override the parent ``position_kp``/``position_kv`` defaults.
    enable_position_servos: bool = False
    position_kp: float = 40.0
    position_kv: float = 12.0
    # Safety clamp applied to every ``send_action``: any joint whose target
    # differs from its current position by more than this magnitude is
    # capped. ``None`` disables clamping.
    max_relative_target: float | dict[str, float] | None = 0.05
    # Treat policy observations/actions as degrees instead of radians.
    use_degrees: bool = False

    # ── Optional live visualization (mirrors ``lerobot.record``'s
    # ``--display_data``/``--display_mode`` flags) ─────────────────────────
    # When any of these are enabled, a :class:`SimRobotVisualizer` will open
    # the corresponding viewer(s) on construction. Set everything to ``False``
    # / ``"off"`` to disable live visualization entirely.
    display_mujoco_viewer: bool = False
    display_camera_windows: bool = False
    # ``"off"``, ``"rerun"`` or ``"foxglove"`` — selects the backend from
    # ``lerobot.utils.visualization_utils`` (rerun spawns a viewer, foxglove
    # starts a WebSocket server on ``display_lerobot_host:display_lerobot_port``).
    display_lerobot_backend: str = "off"
    # For ``"rerun"`` the IP/port point at an optional remote Rerun server.
    # For ``"foxglove"`` the IP is the interface to bind and the port is the
    # WebSocket port.
    display_lerobot_ip: str | None = None
    display_lerobot_port: int | None = None
    display_lerobot_session_name: str = "sim_robot"
    # JPEG-compress images before logging to save bandwidth/CPU.
    display_lerobot_compress_images: bool = False

    def __post_init__(self):
        # Run parent validation (ensures cameras declare width/height/fps).
        super().__post_init__()
        if not self.xml_path:
            raise ValueError("SimRobotConfig.xml_path must point to a MJCF file.")
        if not self.motors:
            raise ValueError(
                "SimRobotConfig.motors must declare at least one motor so the "
                "policy observation/action space is well-defined."
            )
        if self.n_substeps < 1:
            raise ValueError("SimRobotConfig.n_substeps must be >= 1.")
        if self.max_relative_target is not None and not (
            (isinstance(self.max_relative_target, (int, float)) and self.max_relative_target > 0)
            or isinstance(self.max_relative_target, dict)
        ):
            raise ValueError(
                "SimRobotConfig.max_relative_target must be a positive scalar or a "
                "dict of motor_name -> positive scalar, or None to disable."
            )
        if self.default_joint_positions is not None:
            unknown = set(self.default_joint_positions) - set(self.motors)
            if unknown:
                raise ValueError(
                    f"default_joint_positions contains motor names not declared in "
                    f"motors: {sorted(unknown)}."
                )
        if self.display_lerobot_backend not in ("off", "rerun", "foxglove"):
            raise ValueError(
                f"SimRobotConfig.display_lerobot_backend must be one of "
                f"'off', 'rerun', 'foxglove' (got '{self.display_lerobot_backend}')."
            )


# ── Robot implementation ───────────────────────────────────────────────────


class SimRobot(Robot):
    """MuJoCo-backed LeRobot-compatible robot.

    The class is a drop-in replacement for a physical ``Robot`` in any
    LeRobot pipeline: ``get_observation()`` returns proprioceptive keys
    (``"{motor}.pos"``) and rendered image arrays; ``send_action()`` accepts
    a dict of target joint positions and steps the simulator.

    Example:

        >>> cfg = SimRobotConfig(xml_path="so101.xml", motors={...})
        >>> robot = SimRobot(cfg)
        >>> with robot:                                  # auto connect/disconnect
        ...     obs = robot.get_observation()
        ...     robot.send_action({"shoulder_pan.pos": 0.1, ...})
    """

    config_class = SimRobotConfig
    name = "sim_robot"

    # ── Construction ────────────────────────────────────────────────────────

    def __init__(self, config: SimRobotConfig):
        # Sim robots have no per-motor calibration, so we intentionally do
        # NOT call ``super().__init__`` (which would mkdir the default
        # ``~/.cache/huggingface/lerobot/calibration`` tree). Instead, we
        # replicate the small bookkeeping it does and default the directory
        # to a writable tmp location. Users can still opt into the standard
        # cache path via ``calibration_dir=...`` on the config.
        from lerobot.motors import MotorCalibration  # only used by _load_calibration fallback

        self.robot_type = self.name
        self.id = config.id
        if config.calibration_dir is not None:
            self.calibration_dir: Path = Path(config.calibration_dir)
        else:
            tmp_root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
            self.calibration_dir = tmp_root / "lerobot" / "calibration" / "robots" / self.name
        try:
            self.calibration_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"SimRobot: cannot create calibration dir {self.calibration_dir}: {exc}. "
                "Set ``calibration_dir`` on the config to a writable path."
            ) from exc
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()
        self.config = config
        # MuJoCo handles -- populated on ``connect``.
        self._mj_model: Any | None = None
        self._mj_data: Any | None = None
        # Cached lookups.
        self._motor_joint_id: dict[str, int] = {}
        self._motor_actuator_id: dict[str, int] = {}
        self._motor_ctrl_low: dict[str, float] = {}
        self._motor_ctrl_high: dict[str, float] = {}
        self._camera_id: dict[str, int] = {}
        # Offscreen renderers, keyed by (width, height). Public for visualization.
        self._renderers: dict[tuple[int, int], Any] = {}
        self.renderers = self._renderers  # exposed for visualizer read-only access

    # ── Feature schemas ──────────────────────────────────────────────────────

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {f"{m}.pos": float for m in self.config.motors}
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {f"{m}.pos": float for m in self.config.motors}

    # ── Connection state ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._mj_model is not None and self._mj_data is not None

    # Public read-only handles for visualization tooling (e.g. MuJoCo's
    # passive viewer). The returned objects are live; the GUI thread must
    # hold a read/write lock when accessing ``self._mj_data``.
    @property
    def model(self) -> Any:
        return self._mj_model

    @property
    def data(self) -> Any:
        return self._mj_data

    @property
    def is_calibrated(self) -> bool:
        # Simulation has no per-motor calibration to perform.
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Load the MJCF and prepare the MuJoCo simulation.

        Args:
            calibrate: Accepted for API compatibility with the base class;
                simulation robots are always considered calibrated.
        """
        mj = _require_mujoco()
        path = Path(self.config.xml_path)
        if not path.is_file():
            raise FileNotFoundError(f"SimRobot: MJCF model not found at {path}")
        try:
            model = mj.MjModel.from_xml_path(str(path))
        except Exception as e:
            raise RuntimeError(f"SimRobot: failed to load MuJoCo model {path}: {e}") from e
        data = mj.MjData(model)

        # Resolve motor -> joint/actuator ids.
        motor_joint_id: dict[str, int] = {}
        motor_actuator_id: dict[str, int] = {}
        motor_ctrl_low: dict[str, float] = {}
        motor_ctrl_high: dict[str, float] = {}
        for motor_name, spec in self.config.motors.items():
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, spec.joint)
            if jid < 0:
                raise ValueError(
                    f"SimRobot: motor '{motor_name}' refers to joint '{spec.joint}' "
                    f"which was not found in {path}."
                )
            motor_joint_id[motor_name] = jid
            if spec.actuator is not None:
                aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, spec.actuator)
                if aid < 0:
                    raise ValueError(
                        f"SimRobot: motor '{motor_name}' refers to actuator "
                        f"'{spec.actuator}' which was not found in {path}."
                    )
                motor_actuator_id[motor_name] = aid
                lo, hi = model.actuator_ctrlrange[aid].tolist()
                motor_ctrl_low[motor_name] = (
                    float(spec.ctrl_low) if spec.ctrl_low is not None else float(lo)
                )
                motor_ctrl_high[motor_name] = (
                    float(spec.ctrl_high) if spec.ctrl_high is not None else float(hi)
                )

        # Convert generic ``<motor>`` actuators into critically-damped position
        # servos by rewriting ``gainprm[0]=kp`` and ``biasprm[2]=-kv`` on the
        # model in-place. This mirrors what
        # ``assets/SO101/interactive_viewer.py`` does and lets ``SimRobot``
        # drive any plain motor-actuator MJCF without forcing the user to
        # edit the scene by hand. Opt-in via ``SimRobotConfig.enable_position_servos``.
        if self.config.enable_position_servos:
            for motor_name, aid in motor_actuator_id.items():
                spec = self.config.motors[motor_name]
                kp = spec.kp if spec.kp is not None else self.config.position_kp
                kv = spec.kv if spec.kv is not None else self.config.position_kv
                model.actuator_gainprm[aid, 0] = float(kp)
                model.actuator_biasprm[aid, 2] = float(-kv)

        # Resolve camera names.
        camera_id: dict[str, int] = {}
        for cam_name, cam_cfg in self.config.cameras.items():
            cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cam_cfg.name)
            if cid < 0:
                raise ValueError(
                    f"SimRobot: camera '{cam_name}' refers to MuJoCo camera "
                    f"'{cam_cfg.name}' which was not found in {path}."
                )
            camera_id[cam_name] = cid

        self._mj_model = model
        self._mj_data = data
        self._motor_joint_id = motor_joint_id
        self._motor_actuator_id = motor_actuator_id
        self._motor_ctrl_low = motor_ctrl_low
        self._motor_ctrl_high = motor_ctrl_high
        self._camera_id = camera_id
        self._renderers = {}

        # Apply initial state via configure().
        self.configure()
        logger.info("%s connected (nq=%d, nv=%d, nu=%d).", self, model.nq, model.nv, model.nu)

    def configure(self) -> None:
        """Reset the simulator and apply the configured initial state.

        Called once from ``connect()`` and may be called again at any time to
        return the sim to its starting configuration (e.g. between episodes).
        """
        if not self.is_connected:
            return
        mj = _require_mujoco()
        mj.mj_resetData(self._mj_model, self._mj_data)
        if self.config.init_qpos is not None:
            n = min(len(self.config.init_qpos), self._mj_model.nq)
            self._mj_data.qpos[:n] = np.asarray(self.config.init_qpos, dtype=np.float64)
        if self.config.default_joint_positions:
            for motor_name, target in self.config.default_joint_positions.items():
                spec = self.config.motors[motor_name]
                value = spec.initial_position if spec.initial_position is not None else target
                self._set_joint_position(motor_name, self._to_sim_units(float(value)))
        mj.mj_forward(self._mj_model, self._mj_data)

    def calibrate(self) -> None:
        """No-op for simulation; the API requires it to be defined."""
        return None

    # ── Per-step observation and action ─────────────────────────────────────

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        """Read the current proprioception and render configured camera frames."""
        _require_mujoco()  # ensure the dependency is imported before first use
        obs: dict[str, Any] = {}
        for motor_name, jid in self._motor_joint_id.items():
            qpos_adr = int(self._mj_model.jnt_qposadr[jid])
            value = float(self._mj_data.qpos[qpos_adr])
            obs[f"{motor_name}.pos"] = self._from_sim_units(value)
        for cam_name, cid in self._camera_id.items():
            cam_cfg = self.config.cameras[cam_name]
            obs[cam_name] = self._render_camera(cam_name, cam_cfg, cid)
        return obs

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply a position target to each named motor and step the simulation.

        Args:
            action: Mapping of ``"{motor}.pos"`` -> target position. Motors
                omitted from the dict are not commanded (their previous target
                is held) so partial control works naturally.

        Returns:
            The action actually applied, in user units (after safety clamping
            and actuator range clipping).
        """
        mj = _require_mujoco()
        # Resolve which motors are being commanded this step.
        targets: dict[str, float] = {}
        for motor_name in self.config.motors:
            key = f"{motor_name}.pos"
            if key in action:
                raw = action[key]
                if raw is None:
                    continue
                targets[motor_name] = self._to_sim_units(float(raw))

        # Safety clamp: cap per-step magnitude relative to current position.
        if targets and self.config.max_relative_target is not None:
            present = {m: self._read_joint_position(m) for m in targets}
            goal_present = {m: (g, present[m]) for m, g in targets.items()}
            clamped = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            targets = {m: float(v) for m, v in clamped.items()}

        # Apply and step.
        for motor_name, target in targets.items():
            self._set_ctrl_or_pos(motor_name, float(target))
        for _ in range(self.config.n_substeps):
            mj.mj_step(self._mj_model, self._mj_data)

        # Build the response: commanded values where present, current otherwise.
        sent: dict[str, Any] = {}
        for motor_name in self.config.motors:
            if motor_name in targets:
                value = float(targets[motor_name])
            else:
                value = self._read_joint_position(motor_name)
            sent[f"{motor_name}.pos"] = self._from_sim_units(value)
        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        """Release renderers and drop the model/data references."""
        for renderer in self._renderers.values():
            try:
                renderer.close()
            except Exception:  # nosec B110 -- renderer cleanup is best-effort
                pass
        self._renderers.clear()
        self._mj_model = None
        self._mj_data = None
        self._motor_joint_id = {}
        self._motor_actuator_id = {}
        self._motor_ctrl_low = {}
        self._motor_ctrl_high = {}
        self._camera_id = {}
        logger.info("%s disconnected.", self)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _set_ctrl_or_pos(self, motor_name: str, value_rad: float) -> None:
        """Drive the motor: through its actuator when available, else via qpos."""
        if motor_name in self._motor_actuator_id:
            aid = self._motor_actuator_id[motor_name]
            lo = self._motor_ctrl_low[motor_name]
            hi = self._motor_ctrl_high[motor_name]
            self._mj_data.ctrl[aid] = float(np.clip(value_rad, lo, hi))
            return
        self._set_joint_position(motor_name, value_rad)

    def _set_joint_position(self, motor_name: str, value_rad: float) -> None:
        """Set a joint's position directly via ``data.qpos`` (kinematic)."""
        mj = _require_mujoco()
        jid = self._motor_joint_id[motor_name]
        qpos_adr = int(self._mj_model.jnt_qposadr[jid])
        self._mj_data.qpos[qpos_adr] = float(value_rad)
        # Rebuild derived quantities so subsequent renders reflect the change.
        mj.mj_forward(self._mj_model, self._mj_data)

    def _read_joint_position(self, motor_name: str) -> float:
        jid = self._motor_joint_id[motor_name]
        qpos_adr = int(self._mj_model.jnt_qposadr[jid])
        return float(self._mj_data.qpos[qpos_adr])

    def _render_camera(self, cam_name: str, cam_cfg: SimCameraConfig, cam_id: int) -> np.ndarray:
        mj = _require_mujoco()
        key = (cam_cfg.width, cam_cfg.height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mj.Renderer(self._mj_model, height=cam_cfg.height, width=cam_cfg.width)
            self._renderers[key] = renderer
        # mujoco 3.x accepts either an int id or a camera name string.
        renderer.update_scene(self._mj_data, camera=cam_id)
        img = renderer.render()  # ndarray (H, W, 3) uint8 RGB
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        return img

    def _to_sim_units(self, value: float) -> float:
        return float(np.deg2rad(value)) if self.config.use_degrees else float(value)

    def _from_sim_units(self, value: float) -> float:
        return float(np.rad2deg(value)) if self.config.use_degrees else float(value)



# ── Live visualisation ────────────────────────────────────────────────────
# A thin companion to ``SimRobot`` that mirrors ``lerobot.record``'s
# visualisation plumbing:
#   * ``mujoco.viewer.launch_passive`` for an interactive 3D scene,
#   * ``cv2.imshow`` windows for each MuJoCo camera (rendered by the same
#     offscreen renderers ``SimRobot.get_observation`` uses),
#   * ``lerobot.utils.visualization_utils`` for rerun / foxglove streaming.
#
# Construction:
#
#     robot = SimRobot(cfg)
#     with SimRobotVisualizer(robot) as viz:        # auto-wires from cfg.display_*
#         ...
#         obs = robot.get_observation()
#         viz.render(observation=obs)               # repaints windows + logs
#         ...
#
# Or construct explicitly with a different ``display_*`` config than the
# ``SimRobotConfig`` carries, e.g. when running the same sim across viewers.


class SimRobotVisualizer:
    """Live visualisation helper for :class:`SimRobot`.

    Three orthogonal channels can be enabled independently; the constructor
    inspects :attr:`SimRobot.config` (mirroring ``lerobot.record``'s
    ``--display_data``/``--display_mode`` flags) by default and can also be
    driven explicitly:

    =============  ===========================================
    Channel        Triggered when
    =============  ===========================================
    MuJoCo 3D GUI  ``mujoco_viewer=True``  (uses ``mujoco.viewer.launch_passive``)
    Camera windows ``camera_windows=True`` (one ``cv2.namedWindow`` per SimCameraConfig)
    LeRobot stream ``lerobot_backend in {"rerun", "foxglove"}``
    =============  ===========================================

    The LeRobot stream re-uses ``lerobot.utils.visualization_utils`` so its
    wiring is identical to ``lerobot.record --display_data=true
    --display_mode=<mode>``.

    Args:
        robot: A *connected* :class:`SimRobot`. The visualizer reads
            ``robot.model`` and ``robot.data`` for the MuJoCo GUI, and
            ``robot.config.cameras`` to lay out cv2 windows.
        display: Either a :class:`SimRobotConfig` slice (with the same
            ``display_*`` fields) or ``None`` to use ``robot.config`` directly.
            This split lets users reuse the same robot with a different
            visualisation config.

    Raises:
        RuntimeError: if ``robot.is_connected`` is False at construction.
    """

    def __init__(
        self,
        robot: "SimRobot",
        *,
        mujoco_viewer: bool | None = None,
        camera_windows: bool | None = None,
        lerobot_backend: str | None = None,
        lerobot_ip: str | None = None,
        lerobot_port: int | None = None,
        lerobot_session_name: str | None = None,
        lerobot_compress_images: bool | None = None,
        display: Any | None = None,
    ) -> None:
        if not robot.is_connected:
            raise RuntimeError(
                f"{type(self).__name__}: the robot must be connected before a "
                "visualiser is constructed. Call ``robot.connect()`` first."
            )
        cfg = getattr(robot, "config", None)
        # Resolve display flags: explicit arg > `display` object > robot.config.
        def _resolve(name: str, default):
            if display is not None and hasattr(display, name):
                return getattr(display, name)
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
            return default

        self._mujoco_viewer_enabled = bool(
            mujoco_viewer if mujoco_viewer is not None else _resolve("display_mujoco_viewer", False)
        )
        self._camera_windows_enabled = bool(
            camera_windows if camera_windows is not None else _resolve("display_camera_windows", False)
        )
        self._lerobot_backend = (
            lerobot_backend
            if lerobot_backend is not None
            else _resolve("display_lerobot_backend", "off")
        )
        self._lerobot_ip = (
            lerobot_ip if lerobot_ip is not None else _resolve("display_lerobot_ip", None)
        )
        self._lerobot_port = (
            lerobot_port if lerobot_port is not None else _resolve("display_lerobot_port", None)
        )
        self._lerobot_session_name = (
            lerobot_session_name
            if lerobot_session_name is not None
            else _resolve("display_lerobot_session_name", "sim_robot")
        )
        self._lerobot_compress_images = bool(
            lerobot_compress_images
            if lerobot_compress_images is not None
            else _resolve("display_lerobot_compress_images", False)
        )

        if self._lerobot_backend not in ("off",) + tuple(VISUALIZATION_MODES):
            raise ValueError(
                f"Unknown display_lerobot_backend '{self._lerobot_backend}'. "
                f"Expected one of {{'off', *{VISUALIZATION_MODES}}}."
            )

        self._robot = robot
        self._mujoco_handle: Any | None = None
        self._cv2_windows: set[str] = set()
        self._cv2_available = False
        self._started = False
        self._running = True

        # Lazily check cv2 availability even when only the camera windows
        # aren't requested (avoids a hard dep when callers don't use them).
        self._init_visualizers()

    # ── Backend wiring ──────────────────────────────────────────────────────

    def _init_visualizers(self) -> None:
        """Open every enabled backend. Idempotent — safe to call again."""
        if self._started:
            return
        # 1. LeRobot (rerun / foxglove).
        if self._lerobot_backend != "off":
            logger.info(
                "Initialising LeRobot %s visualizer (session=%s, ip=%s, port=%s)",
                self._lerobot_backend,
                self._lerobot_session_name,
                self._lerobot_ip,
                self._lerobot_port,
            )
            # ``init_visualization`` keeps its own static state and is idempotent
            # across visualizer instances -- safer to call only once globally.
            try:
                init_visualization(
                    self._lerobot_backend,
                    session_name=self._lerobot_session_name,
                    ip=self._lerobot_ip,
                    port=self._lerobot_port,
                )
            except Exception as exc:  # nosec B110 - rerun's spawn() needs a display
                logger.warning(
                    "Failed to initialise %s visualizer (%s); continuing without it. "
                    "On a headless host, use --display_lerobot_backend foxglove or "
                    "set --display_lerobot_ip/--display_lerobot_port to a remote Rerun server.",
                    self._lerobot_backend, exc,
                )
                self._lerobot_backend = "off"

        # 2. MuJoCo passive viewer (requires a display server, uses glfw).
        if self._mujoco_viewer_enabled:
            # Glfw's ``__init__`` prints to stderr and calls ``sys.exit(1)``
            # when no display server is reachable -- which would tear the
            # process down before our ``except`` ever runs. Guard up-front so
            # headless hosts (CI, sandboxes, WSL) degrade cleanly.
            has_display = bool(os.environ.get("DISPLAY")) or bool(
                os.environ.get("WAYLAND_DISPLAY")
            )
            if not has_display and sys.platform.startswith("linux"):
                logger.warning(
                    "No DISPLAY/WAYLAND_DISPLAY set; skipping the MuJoCo 3D live "
                    "GUI on this headless host. Other display channels continue."
                )
                self._mujoco_viewer_enabled = False
                self._mujoco_handle = None
            else:
                try:
                    import mujoco.viewer as mj_viewer  # local import to keep deps lean
                except (ImportError, RuntimeError) as exc:
                    logger.warning(
                        "mujoco.viewer unavailable (%s); skipping the 3D live GUI.",
                        exc,
                    )
                    self._mujoco_viewer_enabled = False
                else:
                    try:
                        self._mujoco_handle = mj_viewer.launch_passive(
                            self._robot.model, self._robot.data
                        )
                    except Exception as exc:  # noqa: BLE001 -- backend raises broadly
                        logger.warning(
                            "MuJoCo passive viewer could not start (%s: %s); skipping "
                            "the 3D live GUI. Other display channels continue.",
                            type(exc).__name__, exc,
                        )
                        self._mujoco_handle = None
                        self._mujoco_viewer_enabled = False
                    else:
                        logger.info("MuJoCo passive viewer opened (close the window to stop the loop).")

        # 3. cv2 camera windows.
        if self._camera_windows_enabled:
            try:
                import cv2 as _cv2_probe  # noqa: F401
            except ImportError as exc:
                logger.warning("opencv-python not available (%s); skipping camera windows.", exc)
                self._camera_windows_enabled = False
            else:
                self._cv2_available = True
                for cam_name in self._robot.config.cameras:
                    window_name = f"SimRobot :: {cam_name}"
                    import cv2
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    self._cv2_windows.add(window_name)
                logger.info("Opened %d cv2 camera window(s).", len(self._cv2_windows))

        self._started = True

    # ── Per-step API ───────────────────────────────────────────────────────

    def render(
        self,
        *,
        observation: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
    ) -> None:
        """Push the latest observation and/or action through every enabled backend.

        Call this once per control step *after* ``robot.get_observation`` and
        ``robot.send_action``. Designed to be cheap: nothing is sent if a
        backend was not enabled in the config.
        """
        if not self._started:
            self._init_visualizers()

        # --- MuJoCo 3D viewer -------------------------------------------------
        if self._mujoco_viewer_enabled and self._mujoco_handle is not None:
            if not self._mujoco_handle.is_running():
                # User closed the window -- ask the caller's loop to exit.
                self._running = False
            else:
                self._mujoco_handle.sync()

        # --- cv2 camera windows ----------------------------------------------
        if self._camera_windows_enabled and self._cv2_available and observation:
            try:
                import cv2
            except ImportError:  # already gated above, defensive
                return
            for cam_name, window_name in zip(self._robot.config.cameras, self._cv2_windows):
                if cam_name not in observation:
                    continue
                frame = observation[cam_name]
                if frame is None:
                    continue
                # ``mujoco.Renderer.render`` returns RGB; cv2 expects BGR.
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
                cv2.imshow(window_name, bgr)
            # ``waitKey`` is required for cv2 to actually paint the windows
            # and to keep the window manager responsive (drag, close, ...).
            cv2.waitKey(1)

        # --- LeRobot stream (rerun / foxglove) --------------------------------
        if self._lerobot_backend != "off":
            log_visualization_data(
                self._lerobot_backend,
                observation=observation,
                action=action,
                compress_images=self._lerobot_compress_images,
            )

    @property
    def is_running(self) -> bool:
        """:data:`True` until the user closes the MuJoCo viewer or :meth:`close` is called.

        Use this as the loop predicate in scripts that drive the sim, so closing
        the 3D viewer cleanly stops the simulation.
        """
        if not self._started:
            return True
        if not self._running:
            return False
        if self._mujoco_viewer_enabled and self._mujoco_handle is not None:
            return self._mujoco_handle.is_running()
        return True

    def close(self) -> None:
        """Tear down every backend opened by the constructor.

        Safe to call more than once. ``cv2.destroyAllWindows()`` is gated so it
        runs even if the windows were never created (cv2.destroyAllWindows is
        itself a no-op then, but importing cv2 only when we know it is present
        keeps the dep optional).
        """
        if self._mujoco_handle is not None:
            try:
                self._mujoco_handle.close()
            except Exception:  # nosec B110
                pass
            self._mujoco_handle = None
            self._mujoco_viewer_enabled = False

        if self._cv2_windows:
            try:
                import cv2
                for window_name in list(self._cv2_windows):
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:  # nosec B110
                        pass
                cv2.waitKey(1)
            except ImportError:
                pass
            self._cv2_windows.clear()
            self._camera_windows_enabled = False
            self._cv2_available = False

        if self._lerobot_backend != "off":
            try:
                shutdown_visualization(self._lerobot_backend)
            except Exception:  # nosec B110
                pass
            self._lerobot_backend = "off"

        self._running = False
        self._started = False

    # ── Context-manager helpers ─────────────────────────────────────────────

    def __enter__(self) -> "SimRobotVisualizer":
        if not self._started:
            self._init_visualizers()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()



__all__ = [
    "SimMotorSpec",
    "SimCameraConfig",
    "SimRobotConfig",
    "SimRobot",
    "SimRobotVisualizer",
]
