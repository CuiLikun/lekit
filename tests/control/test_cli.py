from __future__ import annotations

import argparse
import json

import pytest

from lekit.control.cli import _config_values, build_parser, derive_advertise_endpoint
from lekit.robots.piper import PiperCartesianServoConfig, PiperRobotConfig


_PIPER_ROBOT_DOTTED_FIELDS = (
    "channel",
    "speed_percent",
    "max_eef_target_lead_m",
    "max_eef_target_lead_rad",
    "cartesian_servo",
)
_CARTESIAN_SERVO_CONFIG = {
    "position_gain_s": 4.0,
    "orientation_gain_s": 4.0,
    "max_tcp_velocity_m_s": 0.1,
    "max_tcp_angular_velocity_rad_s": 0.5,
    "max_joint_velocity_rad_s": 0.5,
    "max_joint_acceleration_rad_s2": 1.5,
    "max_joint_jerk_rad_s3": 8.0,
    "joint_limit_margin_rad": 0.17453292519943295,
    "jacobian_step_rad": 0.0001,
    "characteristic_length_m": 0.25,
    "singular_value_low": 0.03,
    "singular_value_high": 0.12,
    "minimum_orientation_scale": 0.15,
    "minimum_damping": 0.005,
    "maximum_damping": 0.15,
}


def test_cli_exposes_exact_process_names_and_safe_defaults() -> None:
    parser = build_parser()

    hub = parser.parse_args(["hub"])
    teleop = parser.parse_args(["teleop"])
    robot = parser.parse_args(["robot", "--kind", "piper"])

    assert (hub.command, teleop.command, robot.command) == ("hub", "teleop", "robot")
    assert hub.management_endpoint == "tcp://0.0.0.0:5560"
    assert teleop.action_endpoint == "tcp://0.0.0.0:5557"
    assert robot.enable_motion is False


def test_hub_wildcard_advertisement_requires_and_uses_explicit_host() -> None:
    with pytest.raises(ValueError, match="--advertise-host"):
        derive_advertise_endpoint("tcp://0.0.0.0:5560", None)

    assert derive_advertise_endpoint("tcp://0.0.0.0:5560", "192.168.5.24") == "tcp://192.168.5.24:5560"
    assert derive_advertise_endpoint("tcp://127.0.0.1:5560", None) == "tcp://127.0.0.1:5560"


def test_parser_rejects_unknown_process() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["control-plane"])


def test_subcommands_have_runners() -> None:
    parser = build_parser()
    for command in (["hub"], ["teleop"], ["robot", "--kind", "piper"]):
        args: argparse.Namespace = parser.parse_args(command)
        assert callable(args.runner)


def test_robot_cli_exposes_expected_piper_dotted_fields() -> None:
    parser = build_parser()
    robot = next(action for action in parser._actions if action.dest == "command").choices["robot"]
    options = {option for action in robot._actions for option in action.option_strings}

    assert {f"--robot.{field}" for field in _PIPER_ROBOT_DOTTED_FIELDS} <= options


def test_robot_config_decodes_nested_cartesian_servo_from_json(tmp_path) -> None:
    config_path = tmp_path / "piper.json"
    config_path.write_text(json.dumps({"cartesian_servo": _CARTESIAN_SERVO_CONFIG}), encoding="utf-8")
    args = build_parser().parse_args(
        ["robot", "--kind", "piper", "--robot-config", str(config_path)]
    )

    config = PiperRobotConfig(**_config_values(args, "robot", args.robot_config))

    assert isinstance(config.cartesian_servo, PiperCartesianServoConfig)
    assert config.cartesian_servo == PiperCartesianServoConfig(**_CARTESIAN_SERVO_CONFIG)
