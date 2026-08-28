from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lekit.robots.piper.teleop_processor import (
    PiperIsaacRetargetingStep,
    PiperTeleopProcessorConfig,
    PiperTeleopState,
    make_piper_isaac_processor,
)
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStepRegistry,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

EE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")


def xr_frame(
    *,
    engaged: bool,
    tracked: bool = True,
    translation=(0, 0, 0),
    rotation=(0, 0, 0, 1),
    trigger=0.0,
):
    return {
        "right.translation": np.asarray(translation, dtype=np.float32),
        "right.rotation": np.asarray(rotation, dtype=np.float32),
        "right.trigger": trigger,
        "right.is_tracking": tracked,
        "right.is_engaged": engaged,
    }


def observation(pose=(0.30, -0.10, 0.40, 0.10, -0.20, 0.30)):
    return dict(zip(EE_KEYS, pose, strict=True))


def process(step, action, obs):
    transition = robot_action_observation_to_transition((action, obs))
    return transition_to_robot_action(step(transition))


def engage(step, obs):
    return process(step, xr_frame(engaged=False), obs), process(step, xr_frame(engaged=True), obs)


def test_initial_held_squeeze_stays_unarmed_and_emits_empty_action():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())

    result = process(step, xr_frame(engaged=True), observation())

    assert result == {}
    assert step.state is PiperTeleopState.UNARMED


def test_observing_release_arms_idle():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())

    result = process(step, xr_frame(engaged=False), observation())

    assert result == {}
    assert step.state is PiperTeleopState.IDLE


def test_neutral_engage_latches_same_frame_measured_tcp_without_jump():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation()
    engage(step, obs)

    result = process(step, xr_frame(engaged=True, trigger=0.0), obs)

    assert result.keys() == {*obs, "gripper.pos"}
    for key, value in obs.items():
        assert result[key] == pytest.approx(value)
    assert result["gripper.pos"] == pytest.approx(0.07)
    assert step.state is PiperTeleopState.ENGAGED
    assert step.last_target == result


def test_managed_takeover_uses_nonzero_first_frame_as_operator_anchor():
    config = PiperTeleopProcessorConfig(rotation_scale=1.0)
    step = PiperIsaacRetargetingStep(config)
    obs = observation((0.30, -0.10, 0.40, 0.10, -0.20, 0.30))
    first_translation = np.array([0.012, -0.008, 0.006])
    first_rotation = Rotation.from_euler("xyz", [0.08, -0.04, 0.06])

    step.arm_after_validated_release()
    first = process(
        step,
        xr_frame(
            engaged=True,
            translation=first_translation,
            rotation=first_rotation.as_quat(),
        ),
        obs,
    )

    for key, value in obs.items():
        assert first[key] == pytest.approx(value)
    assert step.state is PiperTeleopState.ENGAGED
    assert step.fault_reason is None

    translation_delta = np.array([0.01, 0.02, 0.03])
    rotation_delta = Rotation.from_euler("xyz", [0.05, -0.03, 0.04])
    second = process(
        step,
        xr_frame(
            engaged=True,
            translation=first_translation + translation_delta,
            rotation=(rotation_delta * first_rotation).as_quat(),
        ),
        obs,
    )

    assert [second[key] for key in EE_KEYS[:3]] == pytest.approx([0.32, -0.11, 0.43])
    operator_to_base = np.asarray(config.operator_to_base_rotation)
    expected_rotation = Rotation.from_matrix(
        operator_to_base @ rotation_delta.as_matrix() @ operator_to_base.T
    ) * Rotation.from_euler("xyz", [obs[key] for key in EE_KEYS[3:]])
    actual_rotation = Rotation.from_euler("xyz", [second[key] for key in EE_KEYS[3:]])
    assert actual_rotation.as_matrix() == pytest.approx(expected_rotation.as_matrix())


def test_cumulative_operator_translation_maps_right_forward_and_up_to_piper_axes():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation((0.0, 0.0, 0.3, 0.0, 0.0, 0.0))
    engage(step, obs)

    result = process(
        step,
        xr_frame(engaged=True, translation=(0.01, 0.02, 0.03)),
        obs,
    )

    assert result["ee.x"] == pytest.approx(0.02)
    assert result["ee.y"] == pytest.approx(-0.01)
    assert result["ee.z"] == pytest.approx(0.33)


def test_quaternion_delta_is_left_composed_in_base_frame():
    config = PiperTeleopProcessorConfig(rotation_scale=1.0)
    step = PiperIsaacRetargetingStep(config)
    obs = observation((0.0, 0.0, 0.3, 0.2, -0.3, 0.4))
    engage(step, obs)
    delta_operator = Rotation.from_euler("xyz", [0.25, -0.10, 0.15])
    action = xr_frame(engaged=True, rotation=delta_operator.as_quat())

    result = process(step, action, obs)

    operator_to_base = np.asarray(config.operator_to_base_rotation)
    expected = Rotation.from_matrix(
        operator_to_base @ delta_operator.as_matrix() @ operator_to_base.T
    ) * Rotation.from_euler("xyz", [obs[key] for key in EE_KEYS[3:]])
    actual = Rotation.from_euler("xyz", [result[key] for key in EE_KEYS[3:]])
    assert actual.as_matrix() == pytest.approx(expected.as_matrix())


def test_release_emits_one_fresh_measured_hold_then_empty():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    old_obs = observation()
    engage(step, old_obs)
    fresh_obs = observation((0.31, -0.08, 0.39, 0.11, -0.19, 0.29))

    hold = process(step, xr_frame(engaged=False), fresh_obs)
    empty = process(step, xr_frame(engaged=False), fresh_obs)

    assert hold == fresh_obs
    assert empty == {}
    assert step.state is PiperTeleopState.IDLE


def test_reengage_latches_new_measured_observation():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    old_obs = observation()
    engage(step, old_obs)
    process(step, xr_frame(engaged=False), old_obs)
    new_obs = observation((0.45, 0.12, 0.35, -0.2, 0.1, -0.4))

    process(step, xr_frame(engaged=False), new_obs)
    result = process(step, xr_frame(engaged=True), new_obs)

    assert result["ee.x"] == pytest.approx(new_obs["ee.x"])
    assert result["ee.y"] == pytest.approx(new_obs["ee.y"])
    assert result["ee.z"] == pytest.approx(new_obs["ee.z"])


def test_tracking_loss_emits_hold_and_requires_release_before_rearm():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation()
    engage(step, obs)
    lost_obs = observation((0.32, -0.06, 0.38, 0.12, -0.18, 0.28))

    hold = process(step, xr_frame(engaged=True, tracked=False), lost_obs)
    blocked = process(step, xr_frame(engaged=True, tracked=True), lost_obs)
    released = process(step, xr_frame(engaged=False, tracked=True), lost_obs)
    reengaged = process(step, xr_frame(engaged=True, tracked=True), lost_obs)

    assert hold == lost_obs
    assert blocked == {}
    assert released == {}
    assert reengaged["ee.x"] == pytest.approx(lost_obs["ee.x"])
    assert step.state is PiperTeleopState.ENGAGED


@pytest.mark.parametrize(
    "bad_action",
    [
        {**xr_frame(engaged=True), "right.translation": np.zeros(2, dtype=np.float32)},
        {**xr_frame(engaged=True), "right.rotation": np.zeros(4, dtype=np.float32)},
        {**xr_frame(engaged=True), "right.trigger": float("nan")},
        {**xr_frame(engaged=True), "right.is_tracking": 1},
    ],
)
def test_malformed_selected_hand_enters_fault_and_emits_one_measured_hold(bad_action):
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation()

    hold = process(step, bad_action, obs)
    assert step.state is PiperTeleopState.FAULT
    empty = process(step, xr_frame(engaged=False), obs)

    assert set(hold) == set(obs)
    for key, value in obs.items():
        assert hold[key] == pytest.approx(value)
    assert empty == {}
    assert step.state is PiperTeleopState.IDLE
    assert step.fault_reason


def test_non_unit_quaternion_enters_fault_and_emits_one_measured_hold():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation()
    process(step, xr_frame(engaged=False), obs)
    bad_action = xr_frame(
        engaged=True,
        rotation=np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
    )

    hold = process(step, bad_action, obs)
    next_result = process(step, xr_frame(engaged=True), obs)

    assert set(hold) == set(obs)
    for key, value in obs.items():
        assert hold[key] == pytest.approx(value)
    assert step.state is PiperTeleopState.FAULT
    assert step.fault_reason is not None
    assert "quaternion unit norm" in step.fault_reason.lower()
    assert next_result == {}


def test_near_unit_quaternion_float_error_is_accepted():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation()
    process(step, xr_frame(engaged=False), obs)
    near_unit = np.array([0.0, 0.0, 0.0, 1.0 + 1e-5], dtype=np.float32)

    result = process(step, xr_frame(engaged=True, rotation=near_unit), obs)

    assert result["ee.x"] == pytest.approx(obs["ee.x"])
    assert result["ee.y"] == pytest.approx(obs["ee.y"])
    assert result["ee.z"] == pytest.approx(obs["ee.z"])
    assert step.state is PiperTeleopState.ENGAGED


def test_malformed_measured_tcp_enters_fault_without_fabricating_a_hold():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    bad_obs = {**observation(), "ee.y": np.asarray([0.0], dtype=np.float32)}

    result = process(step, xr_frame(engaged=False), bad_obs)

    assert result == {}
    assert step.state is PiperTeleopState.FAULT
    assert step.fault_reason


def test_trigger_zero_and_one_map_to_maximum_and_minimum_gripper_width():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation()
    engage(step, obs)

    open_result = process(step, xr_frame(engaged=True, trigger=0.0), obs)
    closed_result = process(step, xr_frame(engaged=True, trigger=1.0), obs)

    assert open_result["gripper.pos"] == pytest.approx(0.07)
    assert closed_result["gripper.pos"] == pytest.approx(0.0)


@pytest.mark.parametrize("trigger_value", [None, "malformed", float("nan")])
def test_arm_only_factory_ignores_missing_or_malformed_trigger_and_omits_gripper(trigger_value):
    pipeline = make_piper_isaac_processor(PiperTeleopProcessorConfig(include_gripper=False))
    obs = observation()
    released = xr_frame(engaged=False)
    engaged = xr_frame(engaged=True)
    if trigger_value is None:
        released.pop("right.trigger")
        engaged.pop("right.trigger")
    else:
        released["right.trigger"] = trigger_value
        engaged["right.trigger"] = trigger_value

    assert pipeline((released, obs)) == {}
    result = pipeline((engaged, obs))

    assert set(result) == set(EE_KEYS)
    assert "gripper.pos" not in result
    assert pipeline.steps[0].state is PiperTeleopState.ENGAGED
    assert pipeline.steps[0].fault_reason is None


def test_reset_clears_state_diagnostics_and_anchor():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    obs = observation()
    engage(step, obs)
    process(step, xr_frame(engaged=True, translation=(0.01, 0, 0)), obs)

    step.reset()

    assert step.state is PiperTeleopState.UNARMED
    assert step.last_target is None
    assert step.fault_reason is None
    assert process(step, xr_frame(engaged=True), obs) == {}


def test_transform_features_removes_all_xr_fields_and_declares_canonical_action_fields():
    raw_action = {
        "left.translation": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        "right.translation": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        "right.rotation": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        "right.trigger": PolicyFeature(type=FeatureType.ACTION, shape=()),
        "right.is_tracking": PolicyFeature(type=FeatureType.ACTION, shape=()),
        "right.is_engaged": PolicyFeature(type=FeatureType.ACTION, shape=()),
    }
    features = {
        PipelineFeatureType.ACTION: raw_action,
        PipelineFeatureType.OBSERVATION: {
            key: PolicyFeature(type=FeatureType.STATE, shape=()) for key in EE_KEYS
        },
    }

    result = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig()).transform_features(features)

    assert set(result[PipelineFeatureType.ACTION]) == {*EE_KEYS, "gripper.pos"}
    assert not any(
        key.startswith("left.") or key.startswith("right.") for key in result[PipelineFeatureType.ACTION]
    )
    assert result[PipelineFeatureType.OBSERVATION] == features[PipelineFeatureType.OBSERVATION]
    assert set(features[PipelineFeatureType.ACTION]) == set(raw_action)


def test_arm_only_transform_features_declares_tcp_without_gripper():
    features = {
        PipelineFeatureType.ACTION: {
            "right.trigger": PolicyFeature(type=FeatureType.ACTION, shape=()),
        },
        PipelineFeatureType.OBSERVATION: {
            key: PolicyFeature(type=FeatureType.STATE, shape=()) for key in EE_KEYS
        },
    }

    result = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig(include_gripper=False)).transform_features(
        features
    )

    assert set(result[PipelineFeatureType.ACTION]) == set(EE_KEYS)
    assert "gripper.pos" not in result[PipelineFeatureType.ACTION]


def test_factory_uses_registered_step_and_official_converters():
    pipeline = make_piper_isaac_processor(PiperTeleopProcessorConfig())

    assert isinstance(pipeline.steps[0], PiperIsaacRetargetingStep)
    assert ProcessorStepRegistry.get("piper_isaac_retargeting") is PiperIsaacRetargetingStep
    assert pipeline.name == "piper_isaac_retargeting"


def test_registered_step_config_can_be_serialized_and_reconstructed():
    config = PiperTeleopProcessorConfig(translation_scale=0.4, rotation_scale=0.2)
    step = PiperIsaacRetargetingStep(config)

    restored = PiperIsaacRetargetingStep(**step.get_config())

    assert restored.config == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hand": "middle"},
        {"translation_scale": float("nan")},
        {"rotation_scale": -1.0},
        {"max_translation_from_anchor_m": 0.0},
        {"max_rotation_from_anchor_rad": -1.0},
        {"neutral_translation_tolerance_m": -1.0},
        {"gripper_min_width_m": 0.07, "gripper_max_width_m": 0.07},
        {"operator_to_base_rotation": ((1, 0, 0), (0, 1, 0), (0, 0, -1))},
    ],
)
def test_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PiperTeleopProcessorConfig(**kwargs)


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_configuration_rejects_non_boolean_include_gripper(value):
    with pytest.raises(ValueError, match="include_gripper"):
        PiperTeleopProcessorConfig(include_gripper=value)


def test_processor_does_not_modify_input_action_or_observation():
    step = PiperIsaacRetargetingStep(PiperTeleopProcessorConfig())
    action = xr_frame(engaged=False)
    obs = observation()
    original_action = deepcopy(action)
    original_obs = deepcopy(obs)

    process(step, action, obs)

    assert action.keys() == original_action.keys()
    for key in action:
        if isinstance(action[key], np.ndarray):
            np.testing.assert_array_equal(action[key], original_action[key])
        else:
            assert action[key] == original_action[key]
    assert obs == original_obs
