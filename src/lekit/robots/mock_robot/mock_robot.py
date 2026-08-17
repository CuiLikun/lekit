"""LeRobot-compatible mock robot for dry-run / CI / pipeline testing.

This module implements :class:`MockRobot`, a drop-in :class:`lerobot.robots.robot.Robot`
subclass that simulates the basic functionality of a real (or simulated) robot
without requiring any hardware, MuJoCo, or ROS. It is intended for:

* Unit / integration tests of policy code that consumes the LeRobot
  ``Robot`` contract.
* Smoke-testing pipelines end-to-end before a physical arm is online.
* Demos and tutorials that want to show the data flow without the
  robotics-specific dependencies of :class:`lekit.robots.sim_robot.SimRobot`
  (mujoco) or :class:`lekit.robots.jaka_robot.JakaRobot` (vendor .so).

The mock keeps a small in-memory state for joint positions. ``send_action``
applies the standard ``max_relative_target`` safety clamp (per-joint scalar
or global float, ``None`` disables it) and moves each joint toward the
clamped target either instantaneously or with first-order smoothing controlled
by ``position_smoothing``. ``get_observation`` returns the current joint
positions (with optional Gaussian noise) and a procedurally generated RGB
frame for every configured camera, so image-typed features behave like a
real camera pipeline.

Example:

    >>> from lekit.robots.mock_robot import MockRobot, MockRobotConfig, MockMotor, MockCamera
    >>> cfg = MockRobotConfig(
    ...     motors={
    ...         "shoulder_pan": MockMotor(id=0, name="shoulder_pan"),
    ...         "shoulder_lift": MockMotor(id=1, name="shoulder_lift"),
    ...     },
    ...     cameras={"front": MockCamera(name="front", width=64, height=48, fps=30)},
    ... )
    >>> robot = MockRobot(cfg)
    >>> with robot:
    ...     obs = robot.get_observation()
    ...     robot.send_action({"shoulder_pan.pos": 0.1, "shoulder_lift.pos": 0.2})
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)


# ── Spec dataclasses ────────────────────────────────────────────────────────


@dataclass
class MockMotor:
    """Lightweight motor descriptor.

    Mirrors the role :class:`lekit.robots.sim_robot.SimMotor` plays for
    :class:`SimRobot`: enough information to address a joint by name
    (``"{name}.pos"``) in observation / action dicts and to look it up by
    stable integer id.
    """

    id: int = -1
    name: str = "motor"
    # Optional symmetric range ``(low, high)``. Used only for documentation
    # and for clamping the initial position; the mock itself does not enforce
    # joint limits on actions (that is what ``max_relative_target`` is for).
    range: tuple[float, float] | None = None


@dataclass
class MockCamera:
    """Camera descriptor.

    Width / height / fps are required by :meth:`RobotConfig.__post_init__`
    whenever a config declares a ``cameras`` dict, so all three are mandatory
    fields here.
    """

    name: str = "camera"
    width: int = 640
    height: int = 480
    fps: int = 30


# ── Config ───────────────────────────────────────────────────────────────────


@RobotConfig.register_subclass("mock_robot")
@dataclass
class MockRobotConfig(RobotConfig):
    """Configuration for :class:`MockRobot`.

    All fields are keyword-only (inherited from ``RobotConfig``). When loaded
    from YAML, ``type: mock_robot`` selects this configuration.
    """

    # motor_name -> MockMotor. The motor name appears as ``"{motor}.pos"``
    # in observation and action dictionaries. If left empty the robot exposes
    # no joints and ``send_action`` / ``get_observation`` carry no proprio.
    motors: dict[str, MockMotor] = field(default_factory=dict)
    # camera_name -> MockCamera. Each entry makes ``get_observation`` emit a
    # procedurally generated ``(H, W, 3)`` uint8 frame under that key.
    cameras: dict[str, MockCamera] = field(default_factory=dict)
    # Optional initial joint positions (motor name -> rad). Joints absent
    # from the dict start at ``0.0``. Overrides the per-motor ``range`` low
    # bound is **not** applied — callers can clamp manually.
    initial_positions: dict[str, float] | None = None
    # Safety clamp applied to every ``send_action``: any joint whose target
    # differs from its current position by more than this magnitude is
    # capped. ``None`` disables clamping.
    max_relative_target: float | dict[str, float] | None = 0.05
    # First-order smoothing applied per ``send_action``: the joint moves from
    # its current position toward the clamped target by this fraction.
    # ``0.0`` means "snap to target" (no dynamics), ``1.0`` means "do not
    # move at all", values in between produce a soft approach. Use a value
    # like ``0.3`` to mimic a sluggish real arm.
    position_smoothing: float = 0.0
    # Standard deviation of zero-mean Gaussian noise added to proprioceptive
    # observations. ``0.0`` (default) makes ``get_observation`` return the
    # exact current positions — useful for unit tests that need determinism.
    observation_noise_std: float = 0.0
    # RNG seed for ``observation_noise_std``. ``None`` lets numpy pick one
    # from the OS entropy source.
    random_seed: int | None = None

    def __post_init__(self):
        # Parent ensures cameras declare width/height/fps.
        super().__post_init__()
        if not 0.0 <= self.position_smoothing < 1.0:
            raise ValueError(
                "MockRobotConfig.position_smoothing must be in [0.0, 1.0) "
                "(1.0 would freeze the robot, use 0.0 for instant snap)."
            )
        if self.observation_noise_std < 0.0:
            raise ValueError("MockRobotConfig.observation_noise_std must be >= 0.")
        if self.max_relative_target is not None and not (
            (isinstance(self.max_relative_target, (int, float)) and self.max_relative_target > 0)
            or isinstance(self.max_relative_target, dict)
        ):
            raise ValueError(
                "MockRobotConfig.max_relative_target must be a positive scalar or a "
                "dict of motor_name -> positive scalar, or None to disable."
            )
        if self.initial_positions is not None:
            unknown = set(self.initial_positions) - set(self.motors)
            if unknown:
                raise ValueError(f"MockRobotConfig.initial_positions references unknown motors: {sorted(unknown)}")


# ── Robot implementation ────────────────────────────────────────────────────


class MockRobot(Robot):
    """In-memory mock that fulfils the LeRobot ``Robot`` contract.

    State management:

    * Joint positions live in ``self._positions`` (motor_name -> float).
    * ``connect`` seeds them from ``config.initial_positions`` (or 0.0).
    * ``send_action`` applies ``ensure_safe_goal_position`` then either
      snaps each joint to the clamped target (``smoothing=0``) or moves a
      fraction ``smoothing`` of the remaining distance toward it.
    * ``get_observation`` returns the current positions (with optional
      Gaussian noise) plus a procedurally generated RGB frame per camera.
      The frame is a slowly-moving colour gradient that encodes the current
      joint positions in its hue/brightness — enough to confirm visually
      that ``send_action`` is plumbed through end-to-end without pulling in
      an image library.

    The class deliberately avoids any external dependency beyond numpy, so
    it can be imported in minimal test environments.
    """

    config_class: ClassVar[type] = MockRobotConfig
    name: ClassVar[str] = "mock_robot"

    # ── Construction ──

    def __init__(self, config: MockRobotConfig):
        super().__init__(config)
        self.config: MockRobotConfig = config
        self.motors: dict[str, MockMotor] = dict(config.motors)
        self.cameras: dict[str, MockCamera] = dict(config.cameras)
        # Per-joint current position. Populated by ``connect``.
        self._positions: dict[str, float] = {}
        # Per-camera frame counter — used to animate the procedural image so
        # successive observations don't return an identical frame.
        self._frame_idx: dict[str, int] = {name: 0 for name in self.cameras}
        # Monotonic clock anchor for the procedural camera animation.
        self._t0: float | None = None
        # Reproducible RNG for observation noise; ``None`` while disconnected.
        self._rng: np.random.Generator | None = None

    # ── Feature schemas ──

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {f"{name}.pos": float for name in self.motors}
        for cam in self.cameras.values():
            features[cam.name] = (cam.height, cam.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {f"{name}.pos": float for name in self.motors}

    # ── Connection state ──

    @property
    def is_connected(self) -> bool:
        return self._t0 is not None

    @property
    def is_calibrated(self) -> bool:
        # The mock has no per-motor calibration step.
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Reset state, seed positions, start the camera clock.

        Args:
            calibrate: Accepted for interface parity — ignored. The mock
                has no calibration to perform.
        """
        del calibrate  # Unused — see docstring.

        seed = self.config.random_seed
        self._rng = np.random.default_rng(seed)
        self._positions = {name: float((self.config.initial_positions or {}).get(name, 0.0)) for name in self.motors}
        for name in self.cameras:
            self._frame_idx[name] = 0
        self._t0 = time.monotonic()
        logger.info(
            "MockRobot[%s] connected (motors=%d, cameras=%d).",
            self.id,
            len(self.motors),
            len(self.cameras),
        )

    @check_if_already_connected
    def configure(self) -> None:
        """No runtime configuration to apply; the mock is stateless."""

    def calibrate(self) -> None:
        """No-op: the mock has nothing to calibrate."""

    @check_if_not_connected
    def disconnect(self) -> None:
        """Drop state so the next ``connect`` starts clean."""
        self._positions.clear()
        self._frame_idx.clear()
        self._t0 = None
        self._rng = None
        logger.info("MockRobot[%s] disconnected.", self.id)

    # ── Observation / Action ──

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return current joint positions (noisy if configured) plus per-camera frames."""
        obs: dict[str, Any] = {}
        rng = self._rng
        std = self.config.observation_noise_std
        for name in self.motors:
            value = float(self._positions[name])
            if std > 0.0 and rng is not None:
                value = float(rng.normal(value, std))
            obs[f"{name}.pos"] = value

        # Wall-clock seconds since ``connect``. Drives the camera animation
        # so frames evolve smoothly between ``get_observation`` calls.
        assert self._t0 is not None  # guaranteed by ``check_if_not_connected``
        t = time.monotonic() - self._t0
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = _render_frame(cam, t, self._positions, self._frame_idx[cam_name])
            self._frame_idx[cam_name] += 1
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Move each named joint toward ``action[name]`` (safety-clamped).

        The clamp uses :func:`ensure_safe_goal_position` so the behaviour
        matches :class:`SimRobot` and :class:`JakaRobot`. After clamping,
        each joint is updated either:

        * snapped to the clamped target when ``position_smoothing == 0``, or
        * moved by ``smoothing`` of the remaining distance otherwise.

        Returns the (clamped, post-smoothing) values actually applied, so
        the caller can log or compare against the raw policy output.
        """
        if not self.motors:
            return {}

        present = {f"{name}.pos": self._positions[name] for name in self.motors}
        goals: dict[str, float] = {}
        for name in self.motors:
            key = f"{name}.pos"
            if key in action:
                try:
                    goals[key] = float(action[key])
                except (TypeError, ValueError) as e:
                    raise TypeError(f"MockRobot.send_action: action[{key!r}]={action[key]!r} is not numeric") from e

        safe: dict[str, float] = {}
        if goals:
            if self.config.max_relative_target is None:
                # Clamp disabled: ``ensure_safe_goal_position`` rejects
                # ``None`` (it only accepts float / dict), which is also
                # why ``JakaRobot`` / ``SimRobot`` never invoke it with
                # ``None`` despite advertising the option in their config.
                safe = dict(goals)
            else:
                safe = ensure_safe_goal_position(
                    {k: (goals[k], present[k]) for k in goals},
                    self.config.max_relative_target,
                )

        smoothing = self.config.position_smoothing
        applied: dict[str, float] = {}
        for name in self.motors:
            key = f"{name}.pos"
            current = self._positions[name]
            if key in safe:
                target = safe[key]
                if smoothing <= 0.0:
                    new_value = target
                else:
                    new_value = current + (target - current) * smoothing
                self._positions[name] = float(new_value)
                applied[key] = float(new_value)
            else:
                applied[key] = current
        return applied


# ── Camera frame synthesis ──────────────────────────────────────────────────


def _render_frame(
    cam: MockCamera,
    t: float,
    positions: dict[str, float],
    frame_idx: int,
) -> np.ndarray:
    """Produce an ``(H, W, 3)`` uint8 RGB frame for the named camera.

    The frame is a smooth colour gradient that:

    * slowly rotates hue over wall-clock time ``t``,
    * uses the current joint positions to perturb hue/saturation per frame,
    * carries a subtle high-frequency dither so the image is not a flat
      block of colour — useful for confirming that a downstream pipeline is
      actually consuming the array rather than caching one frame.

    The output is C-contiguous so it is safe to feed directly to image
    encoders (PNG/JPEG) without an extra copy.
    """
    h, w = cam.height, cam.width
    # Base hue (radians, ``0..2π``) drifts at ~0.05 Hz so the gradient is
    # visibly evolving at a typical 30 Hz control rate.
    base_hue = (t * 0.3 + frame_idx * 0.01) % (2.0 * math.pi)
    # Joint positions push the hue around — gives the frame a deterministic
    # "this is what the arm is doing right now" signature. Clipped to keep
    # the perturbation small relative to the time drift.
    pos_offset = sum(positions.values()) * 0.05
    hue = base_hue + pos_offset

    # Per-pixel coordinates normalised to ``[-1, 1]`` along the height axis.
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    # Vertical gradient maps ``yv`` to a brightness curve; the radial term
    # adds a soft vignette so the centre is brighter than the corners.
    brightness = 0.55 + 0.35 * (1.0 - yv) * 0.5 + 0.10 * (1.0 - xv * xv - yv * yv)

    # Convert HSV-ish (hue, sat=0.55, val=brightness) to RGB. The hue is a
    # single scalar per frame, so the math is just a rotation in the RGB
    # cube — cheap and stable across numpy versions.
    sat = 0.55
    r = brightness * (1.0 + sat * math.cos(hue)) * 0.5
    g = brightness * (1.0 + sat * math.cos(hue - 2.0 * math.pi / 3.0)) * 0.5
    b = brightness * (1.0 + sat * math.cos(hue - 4.0 * math.pi / 3.0)) * 0.5

    frame = np.stack([r, g, b], axis=-1)
    # Low-amplitude deterministic dither breaks up flat regions so the image
    # is visibly noisy even when the arm is stationary.
    dither = (np.sin(xv * 47.0) * np.cos(yv * 31.0) * 2.0)[..., None]
    frame = np.clip((frame + dither / 255.0) * 255.0, 0.0, 255.0).astype(np.uint8)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


__all__ = [
    "MockCamera",
    "MockMotor",
    "MockRobot",
    "MockRobotConfig",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = MockRobotConfig(
        id="demo",
        motors={
            "shoulder_pan": MockMotor(id=0, name="shoulder_pan"),
            "shoulder_lift": MockMotor(id=1, name="shoulder_lift"),
            "elbow": MockMotor(id=2, name="elbow"),
        },
        cameras={"front": MockCamera(name="front", width=64, height=48, fps=30)},
        position_smoothing=0.3,
        observation_noise_std=0.001,
    )
    robot = MockRobot(cfg)
    print(robot)
    with robot:
        for step in range(5):
            time.sleep(0.05)
            action = {f"{n}.pos": 0.1 * (step + 1) for n in robot.motors}
            applied = robot.send_action(action)
            obs = robot.get_observation()
            jp = np.array([obs[f"{n}.pos"] for n in robot.motors])
            print(f"step={step} joints={np.round(jp, 4)} applied={applied}")
