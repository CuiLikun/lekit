"""LeRobot processor for retargeting Isaac XR poses to Piper TCP actions."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from numbers import Real
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    TransitionKey,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.types import RobotAction, RobotObservation

PIPER_EE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
_GRIPPER_KEY = "gripper.pos"
_HAND_FIELDS = ("translation", "rotation", "trigger", "is_tracking", "is_engaged")


class PiperTeleopState(StrEnum):
    """Safety state of the Isaac-to-Piper clutch."""

    UNARMED = "unarmed"
    IDLE = "idle"
    ENGAGED = "engaged"
    FAULT = "fault"


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must be a numpy array with shape {shape}")
    if (
        not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.bool_)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must have a numeric dtype")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


@dataclass
class PiperTeleopProcessorConfig:
    """Validated coordinate mapping, clutch, motion, and gripper settings."""

    hand: str = "right"
    include_gripper: bool = True
    translation_scale: float = 1.0
    rotation_scale: float = 0.0
    operator_to_base_rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    max_translation_from_anchor_m: float = 0.15
    max_rotation_from_anchor_rad: float = math.radians(30.0)
    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float = 0.07
    neutral_translation_tolerance_m: float = 1e-4
    neutral_rotation_tolerance_rad: float = math.radians(0.5)

    def __post_init__(self) -> None:
        if self.hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")
        if not isinstance(self.include_gripper, bool):
            raise ValueError("include_gripper must be a boolean")

        for name in ("translation_scale", "rotation_scale"):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, value)

        for name in ("max_translation_from_anchor_m", "max_rotation_from_anchor_rad"):
            value = _finite_real(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)

        minimum = _finite_real(self.gripper_min_width_m, "gripper_min_width_m")
        maximum = _finite_real(self.gripper_max_width_m, "gripper_max_width_m")
        if minimum < 0.0 or minimum >= maximum:
            raise ValueError("gripper widths must be finite, non-negative, and increasing")
        self.gripper_min_width_m = minimum
        self.gripper_max_width_m = maximum

        for name in ("neutral_translation_tolerance_m", "neutral_rotation_tolerance_rad"):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, value)

        try:
            matrix = np.asarray(self.operator_to_base_rotation, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("operator_to_base_rotation must be a finite 3x3 matrix") from exc
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("operator_to_base_rotation must be a finite 3x3 matrix")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=1e-6):
            raise ValueError("operator_to_base_rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("operator_to_base_rotation must have determinant +1")
        self.operator_to_base_rotation = tuple(tuple(float(item) for item in row) for row in matrix)


@ProcessorStepRegistry.register("piper_isaac_retargeting")
@dataclass
class PiperIsaacRetargetingStep(RobotActionProcessorStep):
    """Convert cumulative engage-relative Isaac poses to absolute Piper TCP targets."""

    config: PiperTeleopProcessorConfig = field(default_factory=PiperTeleopProcessorConfig)
    _state: PiperTeleopState = field(init=False, default=PiperTeleopState.UNARMED)
    _anchor_position: np.ndarray | None = field(init=False, default=None, repr=False)
    _anchor_rotation: Rotation | None = field(init=False, default=None, repr=False)
    _operator_anchor_position: np.ndarray | None = field(init=False, default=None, repr=False)
    _operator_anchor_rotation: Rotation | None = field(init=False, default=None, repr=False)
    _validated_release: bool = field(init=False, default=False, repr=False)
    _requires_release: bool = field(init=False, default=False, repr=False)
    _last_target: RobotAction | None = field(init=False, default=None, repr=False)
    _fault_reason: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.config, dict):
            self.config = PiperTeleopProcessorConfig(**self.config)
        elif not isinstance(self.config, PiperTeleopProcessorConfig):
            raise TypeError("config must be a PiperTeleopProcessorConfig")

    @property
    def state(self) -> PiperTeleopState:
        return self._state

    @property
    def last_target(self) -> RobotAction | None:
        return None if self._last_target is None else dict(self._last_target)

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason

    def get_config(self) -> dict[str, Any]:
        return {"config": asdict(self.config)}

    def reset(self) -> None:
        self._state = PiperTeleopState.UNARMED
        self._anchor_position = None
        self._anchor_rotation = None
        self._operator_anchor_position = None
        self._operator_anchor_rotation = None
        self._validated_release = False
        self._requires_release = False
        self._last_target = None
        self._fault_reason = None

    def arm_after_validated_release(self) -> None:
        """Arm a fresh stream after its authority layer observed a release."""

        if self._state is not PiperTeleopState.UNARMED:
            raise RuntimeError("validated release can only arm an unarmed processor")
        self._state = PiperTeleopState.IDLE
        self._validated_release = True

    def action(self, action: RobotAction) -> RobotAction:
        try:
            observation = self.transition[TransitionKey.OBSERVATION]
        except KeyError:
            observation = None
        measured, observation_error = self._read_measured_tcp(observation)
        hand, hand_error = self._read_hand(action)
        if observation_error is not None or hand_error is not None:
            reason = observation_error or hand_error or "invalid teleoperation input"
            return self._fault(reason, measured)
        assert measured is not None and hand is not None

        if self._state is PiperTeleopState.FAULT:
            if hand["is_tracking"] and not hand["is_engaged"]:
                self._state = PiperTeleopState.IDLE
                self._requires_release = False
                self._clear_anchor()
            return {}

        tracked = hand["is_tracking"]
        engaged = hand["is_engaged"]
        if self._state is PiperTeleopState.UNARMED:
            if tracked and not engaged:
                self._state = PiperTeleopState.IDLE
            return {}

        if self._state is PiperTeleopState.IDLE:
            if self._requires_release:
                if tracked and not engaged:
                    self._requires_release = False
                return {}
            if not tracked:
                self._validated_release = False
                self._requires_release = True
                return {}
            if not engaged:
                return {}
            if not self._validated_release and not self._is_neutral(
                hand["translation"], hand["rotation"]
            ):
                return self._fault("engage frame is not neutral", measured)
            self._anchor_position = measured[:3].copy()
            self._anchor_rotation = Rotation.from_euler("xyz", measured[3:])
            self._operator_anchor_position = hand["translation"].copy()
            self._operator_anchor_rotation = Rotation.from_quat(hand["rotation"])
            self._validated_release = False
            self._state = PiperTeleopState.ENGAGED
            return self._target_action(hand, measured)

        assert self._state is PiperTeleopState.ENGAGED
        if not tracked:
            self._state = PiperTeleopState.IDLE
            self._requires_release = True
            self._clear_anchor()
            return self._hold(measured)
        if not engaged:
            self._state = PiperTeleopState.IDLE
            self._clear_anchor()
            return self._hold(measured)
        return self._target_action(hand, measured)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = deepcopy(features)
        action_keys = (*PIPER_EE_KEYS, _GRIPPER_KEY) if self.config.include_gripper else PIPER_EE_KEYS
        transformed[PipelineFeatureType.ACTION] = {
            key: PolicyFeature(type=FeatureType.ACTION, shape=()) for key in action_keys
        }
        return transformed

    def _target_action(self, hand: dict[str, Any], measured: np.ndarray) -> RobotAction:
        assert (
            self._anchor_position is not None
            and self._anchor_rotation is not None
            and self._operator_anchor_position is not None
            and self._operator_anchor_rotation is not None
        )
        delta_position = self.config.translation_scale * (
            hand["translation"] - self._operator_anchor_position
        )
        delta_position = self._clamp_norm(delta_position, self.config.max_translation_from_anchor_m)
        base_delta = np.asarray(self.config.operator_to_base_rotation) @ delta_position
        target_position = self._anchor_position + base_delta

        delta_operator = Rotation.from_quat(hand["rotation"]) * self._operator_anchor_rotation.inv()
        delta_rotvec = self._clamp_norm(
            delta_operator.as_rotvec() * self.config.rotation_scale,
            self.config.max_rotation_from_anchor_rad,
        )
        delta_operator = Rotation.from_rotvec(delta_rotvec)
        operator_to_base = np.asarray(self.config.operator_to_base_rotation)
        delta_base = Rotation.from_matrix(operator_to_base @ delta_operator.as_matrix() @ operator_to_base.T)
        target_rotation = delta_base * self._anchor_rotation
        target_pose = np.concatenate((target_position, target_rotation.as_euler("xyz")))
        result = dict(zip(PIPER_EE_KEYS, (float(value) for value in target_pose), strict=True))
        if self.config.include_gripper:
            result[_GRIPPER_KEY] = self._trigger_to_width(hand["trigger"])
        self._last_target = dict(result)
        return result

    def _fault(self, reason: str, measured: np.ndarray | None) -> RobotAction:
        was_fault = self._state is PiperTeleopState.FAULT
        self._state = PiperTeleopState.FAULT
        self._requires_release = True
        self._clear_anchor()
        self._fault_reason = reason
        if was_fault or measured is None:
            return {}
        return self._hold(measured)

    def _hold(self, measured: np.ndarray) -> RobotAction:
        result = dict(zip(PIPER_EE_KEYS, (float(value) for value in measured), strict=True))
        self._last_target = dict(result)
        return result

    def _read_measured_tcp(self, observation: Any) -> tuple[np.ndarray | None, str | None]:
        if not isinstance(observation, dict):
            return None, "observation is missing or not a mapping"
        values: list[float] = []
        try:
            for key in PIPER_EE_KEYS:
                values.append(_finite_real(observation[key], f"observation[{key!r}]"))
        except (KeyError, ValueError) as exc:
            return None, str(exc)
        return np.asarray(values, dtype=float), None

    def _read_hand(self, action: RobotAction) -> tuple[dict[str, Any] | None, str | None]:
        prefix = f"{self.config.hand}."
        required_fields = (
            _HAND_FIELDS
            if self.config.include_gripper
            else tuple(field for field in _HAND_FIELDS if field != "trigger")
        )
        missing = [f"{prefix}{field}" for field in required_fields if f"{prefix}{field}" not in action]
        if missing:
            return None, f"selected hand fields are missing: {missing}"
        try:
            translation = _finite_array(action[f"{prefix}translation"], (3,), "hand translation")
            rotation = _finite_array(action[f"{prefix}rotation"], (4,), "hand rotation")
            norm = float(np.linalg.norm(rotation))
            if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("hand rotation quaternion unit norm must be 1")
            rotation /= norm
            if self.config.include_gripper:
                trigger = _finite_real(action[f"{prefix}trigger"], "hand trigger")
                if not 0.0 <= trigger <= 1.0:
                    raise ValueError("hand trigger must be in [0, 1]")
            tracking = action[f"{prefix}is_tracking"]
            engaged = action[f"{prefix}is_engaged"]
            if not isinstance(tracking, (bool, np.bool_)):
                raise ValueError("hand is_tracking must be a boolean")
            if not isinstance(engaged, (bool, np.bool_)):
                raise ValueError("hand is_engaged must be a boolean")
        except (TypeError, ValueError) as exc:
            return None, str(exc)
        result = {
            "translation": translation,
            "rotation": rotation,
            "is_tracking": bool(tracking),
            "is_engaged": bool(engaged),
        }
        if self.config.include_gripper:
            result["trigger"] = trigger
        return result, None

    def _is_neutral(self, translation: np.ndarray, rotation: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(translation) <= self.config.neutral_translation_tolerance_m
            and Rotation.from_quat(rotation).magnitude() <= self.config.neutral_rotation_tolerance_rad
        )

    def _trigger_to_width(self, trigger: float) -> float:
        return float(
            self.config.gripper_max_width_m
            - trigger * (self.config.gripper_max_width_m - self.config.gripper_min_width_m)
        )

    @staticmethod
    def _clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= maximum or norm <= 1e-12:
            return vector
        return vector * (maximum / norm)

    def _clear_anchor(self) -> None:
        self._anchor_position = None
        self._anchor_rotation = None
        self._operator_anchor_position = None
        self._operator_anchor_rotation = None


def make_piper_isaac_processor(
    config: PiperTeleopProcessorConfig | None = None,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Build the registered Piper Isaac retargeting pipeline."""

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[PiperIsaacRetargetingStep(config=config or PiperTeleopProcessorConfig())],
        name="piper_isaac_retargeting",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


__all__ = [
    "PIPER_EE_KEYS",
    "PiperIsaacRetargetingStep",
    "PiperTeleopProcessorConfig",
    "PiperTeleopState",
    "make_piper_isaac_processor",
]
