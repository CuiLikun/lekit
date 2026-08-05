import pytest

from lerobot.configs.dataset import DatasetRecordConfig
from robots.jaka_robot import JakaRobotConfig

from . import record as record_module
from .record import RecordConfig
from .xr import XRControllerConfig


class _BuildStoppedError(Exception):
    pass


def test_live_recording_preserves_power_and_enable(monkeypatch):
    cfg = RecordConfig(
        robot=JakaRobotConfig(auto_power_on=True, auto_enable=True),
        teleop=XRControllerConfig(),
        dataset=DatasetRecordConfig(fps=30),
    )
    live_flags = None

    def capture_before_hardware(actual_cfg):
        nonlocal live_flags
        live_flags = (actual_cfg.robot.auto_power_on, actual_cfg.robot.auto_enable)
        raise _BuildStoppedError

    monkeypatch.setattr(record_module, "build_device", capture_before_hardware)

    with pytest.raises(_BuildStoppedError):
        record_module.record.__wrapped__(cfg)

    assert live_flags == (True, True)
