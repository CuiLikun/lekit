from __future__ import annotations

import argparse

import pytest

from lekit.control.cli import build_parser, derive_advertise_endpoint


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
