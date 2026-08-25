"""Session lifecycle tests that do not require XR hardware."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from lekit.teleoperators.isaac_teleop.config import IsaacTeleopConfig
from lekit.teleoperators.isaac_teleop.xr_controller import IsaacXRController


class _FakeTeleopSession:
    def __init__(self, enter_error: BaseException | None = None):
        self.enter_error = enter_error
        self.exited = False

    def __enter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.exited = True


class _FakeLauncher:
    def __init__(self):
        self.health_calls = 0
        self.stop_calls = 0

    def health_check(self) -> None:
        self.health_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _controller(config: IsaacTeleopConfig | None = None) -> IsaacXRController:
    controller = object.__new__(IsaacXRController)
    controller.config = config or IsaacTeleopConfig()
    controller._session = None
    controller._cloudxr_launcher = _FakeLauncher()
    controller.calibrate = lambda: None
    controller._build_pipeline = lambda: object()
    controller._ensure_cloudxr_runtime = lambda: None
    return controller


def test_connect_waits_for_form_factor_without_stopping_cloudxr(monkeypatch: pytest.MonkeyPatch) -> None:
    from lekit.teleoperators.isaac_teleop import session as session_module

    attempts = iter(
        [
            _FakeTeleopSession(RuntimeError("Failed to get OpenXR system: -35")),
            _FakeTeleopSession(),
        ]
    )
    monkeypatch.setattr(session_module, "TeleopSessionConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(session_module, "TeleopSession", lambda _config: next(attempts))
    monkeypatch.setattr(session_module.time, "sleep", lambda _seconds: None)

    controller = _controller()
    wait_notifications = []
    controller.set_connect_wait_callback(lambda: wait_notifications.append("waiting"))
    launcher = controller._cloudxr_launcher
    controller.connect()

    assert controller.is_connected
    assert wait_notifications == ["waiting"]
    assert controller._cloudxr_launcher is launcher
    assert launcher.health_calls == 1
    assert launcher.stop_calls == 0


def test_connect_suppresses_native_openxr_logs_after_first_wait_attempt(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from lekit.teleoperators.isaac_teleop import session as session_module

    attempts = 0

    class _NativeLoggingSession(_FakeTeleopSession):
        def __enter__(self):
            nonlocal attempts
            attempts += 1
            os.write(1, b"Created OpenXR instance\n")
            os.write(2, b"LOG in xrCreateInstance: Instance created\n")
            if attempts < 3:
                raise RuntimeError("Failed to get OpenXR system: -35")
            return self

    monkeypatch.setattr(session_module, "TeleopSessionConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(session_module, "TeleopSession", _NativeLoggingSession)
    monkeypatch.setattr(session_module.time, "sleep", lambda _seconds: None)

    controller = _controller()
    controller.connect()

    captured = capfd.readouterr()
    assert attempts == 3
    assert captured.out.count("Created OpenXR instance") == 1
    assert captured.err.count("LOG in xrCreateInstance") == 1


def test_connect_does_not_retry_unrelated_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from lekit.teleoperators.isaac_teleop import session as session_module

    calls = 0

    def create_session(_config):
        nonlocal calls
        calls += 1
        return _FakeTeleopSession(RuntimeError("Vulkan initialization failed"))

    monkeypatch.setattr(session_module, "TeleopSessionConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(session_module, "TeleopSession", create_session)

    controller = _controller()
    launcher = controller._cloudxr_launcher
    with pytest.raises(RuntimeError, match="Vulkan initialization failed"):
        controller.connect()

    assert calls == 1
    assert launcher.stop_calls == 1
    assert controller._cloudxr_launcher is None


def test_connect_timeout_stops_cloudxr(monkeypatch: pytest.MonkeyPatch) -> None:
    from lekit.teleoperators.isaac_teleop import session as session_module

    clock = SimpleNamespace(now=0.0)
    calls = 0

    def create_session(_config):
        nonlocal calls
        calls += 1
        return _FakeTeleopSession(RuntimeError("Failed to get OpenXR system: -35"))

    def sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(session_module, "TeleopSessionConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(session_module, "TeleopSession", create_session)
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(session_module.time, "sleep", sleep)

    controller = _controller(IsaacTeleopConfig(connect_timeout_s=1.0, connect_retry_interval_s=0.4))
    launcher = controller._cloudxr_launcher
    with pytest.raises(TimeoutError, match="within 1 seconds"):
        controller.connect()

    assert calls == 4
    assert launcher.stop_calls == 1
    assert controller._cloudxr_launcher is None
