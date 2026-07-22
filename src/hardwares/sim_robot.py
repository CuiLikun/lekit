from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

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
                "mujoco is required to use SimRobot. Install it with `pip install mujoco` or `uv pip install mujoco`."
            ) from e
        _MUJOCO = _m
    return _MUJOCO


# ── Configs ─────────────────────────────────────────────────────────────────


@dataclass
class SimMotorSpec:
    id: int = -1
    name: str = "Motor X"
    gear: float = 1.0  # Gear ratio
    gain: float = 1.0  # Proportional gain for position servo
    bias: float = 0.0  # Bias for position servo
    ctrl: tuple[float, float] | None = None  # (low, high) control range


@dataclass
class SimCameraConfig:
    id: int = -1
    name: str = "Camera X"
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
    # Safety clamp applied to every ``send_action``: any joint whose target
    # differs from its current position by more than this magnitude is
    # capped. ``None`` disables clamping.
    max_relative_target: float | dict[str, float] | None = 0.05
    # Treat policy observations/actions as degrees instead of radians.
    use_degrees: bool = False

    def __post_init__(self):
        # Run parent validation (ensures cameras declare width/height/fps).
        super().__post_init__()
        if not Path(self.xml_path).exists():
            raise ValueError(f"SimRobotConfig.xml_path ({self.xml_path}) does not exist.")
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
                    f"default_joint_positions contains motor names not declared in motors: {sorted(unknown)}."
                )
        self.introspect_scene()

    def introspect_scene(self) -> dict:
        """Load the MJCF and return auto-derived motor/actuator/camera specs."""
        mj = _require_mujoco()
        model = mj.MjModel.from_xml_path(self.xml_path)
        # Auto-derive motor specs and camera specs.
        self.motors = {}
        for i in range(model.nu):
            actuator = model.actuator(i)
            # Exclude non-joint actuators
            if actuator.trntype != 0 or actuator.trnid[0] == -1:
                continue
            joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, actuator.trnid[0])
            # Exclude actuators with different joint names
            if joint_name is None or joint_name != actuator.name:
                continue
            # Construct a SimMotorSpec for this actuator.
            self.motors[actuator.name] = SimMotorSpec(
                id=i,
                name=actuator.name,
                gear=actuator.gear,
                gain=actuator.gainprm,
                bias=actuator.biasprm,
                ctrl=actuator.ctrlrange,
            )
        self.cameras.update(
            {
                (item := model.cam(i)).name: SimCameraConfig(
                    id=i,
                    name=item.name,
                    width=640,
                    height=480,
                    fps=30,
                )
                for i in range(model.ncam)
            }
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
        self.config = config
        self.motors = config.motors
        self.cameras = config.cameras
        # MuJoCo handles -- populated on ``connect``.
        self.mj_model: Any | None = None
        self.mj_data: Any | None = None
        # Offscreen renderers, keyed by (width, height); populated on ``connect``.
        self.renderers: dict[tuple[int, int], Any] = {}
        # ``mujoco`` module handle; populated in ``connect()`` via the
        # lazy-import helper so methods can call ``self.mj.X(...)`` directly.
        self.mj: Any | None = None

    # ── Feature schemas ──────────────────────────────────────────────────────

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {f"{m.name}.pos": float for m in self.motors.values()}
        for name, cam in self.cameras.items():
            features[name] = (cam.height, cam.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {f"{m.name}.pos": float for m in self.motors.values()}

    # ── Connection state ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self.mj_model is not None and self.mj_data is not None

    # Public read-only handles for the live MuJoCo model/data. The returned
    # objects are live and may be mutated by the simulation step.
    @property
    def model(self) -> Any:
        return self.mj_model

    @property
    def data(self) -> Any:
        return self.mj_data

    @property
    def is_calibrated(self) -> bool:
        # Simulation has no per-motor calibration to perform.
        return True

    @check_if_already_connected
    def connect(self) -> None:
        """Load the MJCF and prepare the MuJoCo simulation."""
        self.mj = _require_mujoco()
        path = Path(self.config.xml_path)
        if not path.is_file():
            raise FileNotFoundError(f"SimRobot: MJCF model not found at {path}")
        try:
            self.mj_model = self.mj.MjModel.from_xml_path(str(path))
        except Exception as e:
            raise RuntimeError(f"SimRobot: failed to load MuJoCo model {path}: {e}") from e
        self.mj_data = self.mj.MjData(self.mj_model)

        # Apply initial state via configure().
        self.configure()

        # Pre-initialize an offscreen renderer for each configured camera so
        # the first ``get_observation`` doesn't pay the allocation cost. The
        # cache is keyed by (width, height); cameras sharing dimensions reuse
        # the same renderer.
        for cam in self.cameras.values():
            key = (cam.width, cam.height)
            if key not in self.renderers:
                self.renderers[key] = self.mj.Renderer(self.mj_model, height=cam.height, width=cam.width)

        logger.info(
            "%s connected (nq=%d, nv=%d, nu=%d).", self.name, self.mj_model.nq, self.mj_model.nv, self.mj_model.nu
        )

    def configure(self) -> None:
        """Reset the simulator and apply the configured initial state.

        Called once from ``connect()`` and may be called again at any time to
        return the sim to its starting configuration (e.g. between episodes).
        """
        if not self.is_connected:
            return
        self.mj.mj_resetData(self.mj_model, self.mj_data)
        if self.config.init_qpos is not None:
            n = min(len(self.config.init_qpos), self.mj_model.nq)
            self.mj_data.qpos[:n] = np.asarray(self.config.init_qpos, dtype=np.float64)
        if self.config.default_joint_positions:
            for motor_name, target in self.config.default_joint_positions.items():
                spec = self.config.motors[motor_name]
                value = spec.initial_position if spec.initial_position is not None else target
                self._apply_motor_target(motor_name, self._to_sim_units(float(value)))
        self.mj.mj_forward(self.mj_model, self.mj_data)

    def calibrate(self) -> None:
        """No-op for simulation; the API requires it to be defined."""
        return None

    # ── Per-step observation and action ─────────────────────────────────────

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read the current proprioception and render configured camera frames."""
        obs: dict[str, Any] = {}
        for name in self.motors:
            obs[f"{name}.pos"] = self.mj_data.joint(name).qpos
        for name, cam in self.cameras.items():
            renderer = self.renderers[(cam.width, cam.height)]
            renderer.update_scene(self.mj_data, camera=cam.id)
            img = renderer.render()  # (H, W, 3) uint8 RGB
            if not img.flags["C_CONTIGUOUS"]:
                img = np.ascontiguousarray(img)
            obs[name] = img
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Apply a position target to each named motor and step the simulation.

        Args:
            action: Mapping of ``"{motor}.pos"`` -> target position. Motors
                omitted from the dict are not commanded (their previous target
                is held) so partial control works naturally.

        Returns:
            The action actually applied, in user units (after safety clamping
            and actuator range clipping).
        """
        # Resolve which motors are being commanded this step.
        targets: dict[str, float] = {}
        for motor_name in self.config.motors:
            raw = action.get(f"{motor_name}.pos")
            if raw is None:
                continue
            targets[motor_name] = self._to_sim_units(float(raw))

        # Safety clamp: cap per-step magnitude relative to current position.
        if targets and self.config.max_relative_target is not None:
            present = {m: self._read_motor_position(m) for m in targets}
            goal_present = {m: (g, present[m]) for m, g in targets.items()}
            clamped = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            targets = {m: float(v) for m, v in clamped.items()}

        # Apply and step.
        for motor_name, target in targets.items():
            self._apply_motor_target(motor_name, float(target))
        for _ in range(self.config.n_substeps):
            self.mj.mj_step(self.mj_model, self.mj_data)

        # Build the response: commanded values where present, current otherwise.
        sent: dict[str, Any] = {}
        for motor_name in self.config.motors:
            if motor_name in targets:
                value = float(targets[motor_name])
            else:
                value = self._read_motor_position(motor_name)
            sent[f"{motor_name}.pos"] = self._from_sim_units(value)
        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        """Release renderers and drop the model/data references."""
        for renderer in self.renderers.values():
            try:
                renderer.close()
            except Exception:  # nosec B110 -- renderer cleanup is best-effort
                pass
        self.renderers.clear()
        self.mj_model = None
        self.mj_data = None
        logger.info("%s disconnected.", self)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _apply_motor_target(self, motor_name: str, value_rad: float) -> None:
        """Drive ``motor_name`` to ``value_rad`` — through its actuator when one
        exists, else by writing ``data.qpos`` directly. Both lookups are by
        name and resolved on each call, so they tolerate joint/actuator ids
        changing between connect/reconnect cycles.
        """
        aid = self.mj.mj_name2id(self.mj_model, self.mj.mjtObj.mjOBJ_ACTUATOR, motor_name)
        if aid >= 0:
            lo, hi = self.mj_model.actuator_ctrlrange[aid]
            self.mj_data.ctrl[aid] = float(np.clip(value_rad, lo, hi))
            return
        jid = self.mj.mj_name2id(self.mj_model, self.mj.mjtObj.mjOBJ_JOINT, motor_name)
        qpos_adr = int(self.mj_model.jnt_qposadr[jid])
        self.mj_data.qpos[qpos_adr] = float(value_rad)
        # Rebuild derived quantities so subsequent renders reflect the change.
        self.mj.mj_forward(self.mj_model, self.mj_data)

    def _read_motor_position(self, motor_name: str) -> float:
        jid = self.mj.mj_name2id(self.mj_model, self.mj.mjtObj.mjOBJ_JOINT, motor_name)
        qpos_adr = int(self.mj_model.jnt_qposadr[jid])
        return float(self.mj_data.qpos[qpos_adr])

    def _to_sim_units(self, value: float) -> float:
        return float(np.deg2rad(value)) if self.config.use_degrees else float(value)

    def _from_sim_units(self, value: float) -> float:
        return float(np.rad2deg(value)) if self.config.use_degrees else float(value)


__all__ = [
    "SimMotorSpec",
    "SimCameraConfig",
    "SimRobotConfig",
    "SimRobot",
]

if __name__ == "__main__":
    cfg = SimRobotConfig(xml_path="/home/sorel/workspace/avatar/assets/SO101/scene.xml")
    robot = SimRobot(cfg)

    with robot:
        obs = robot.get_observation()
        print(obs.keys())
