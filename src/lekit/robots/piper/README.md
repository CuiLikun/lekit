# PiperRobot

`PiperRobot` adapts the AgileX `pyAgxArm` driver to LeRobot 0.6. Joint positions use radians and
`gripper.pos` uses metres.

Install the optional hardware dependency:

```bash
uv sync --extra piper
```

Basic usage:

```python
from lekit.robots.piper import PiperRobot, PiperRobotConfig

robot = PiperRobot(
    PiperRobotConfig(
        channel="can0",
        firmware_version="default",  # select v183/v188/v189 for matching controller firmware
        cameras={},
    )
)

with robot:
    observation = robot.get_observation()
    hold_action = {key: observation[key] for key in robot.action_features}
    applied = robot.send_action(hold_action)
```

## Self-test demo

Run the formatted static self-test first. It reads communication, firmware, controller, six-axis,
flange, and gripper diagnostics without enabling or moving the arm:

```bash
uv run python -m lekit.robots.piper.demo --firmware-version v188
```

Select the firmware driver reported by the static check. To add operator-confirmed 10% speed motion
checks, where each joint moves by 0.01 rad and returns to its refreshed starting position, use:

```bash
uv run python -m lekit.robots.piper.demo --firmware-version v188 --full
```

The dynamic phase is blocked by static safety failures. An unhomed gripper is reported and skipped
instead of being calibrated or moved automatically.

## Hardware validation

Before sending a new target on a physical arm:

1. Confirm the CAN interface is up at 1 Mbps and the arm is mechanically supported.
2. Connect once with `auto_enable=False`, then verify `get_observation()` returns six plausible joint
   values and a plausible gripper width.
3. Verify `robot.arm.get_joint_limits_enabled()` is `True`.
4. Reconnect with `speed_percent=10`, read the current observation, filter it to
   `{key: observation[key] for key in robot.action_features}`, and pass that unchanged state to
   `send_action()`. Confirm measured drift remains within the site's safety tolerance.
5. Only then test a small bounded target. Keep `max_relative_target` enabled and an emergency stop within
   reach.

`disconnect()` intentionally leaves the joints enabled by default because disabling a raised Piper arm can
make it fall. Set `disable_on_disconnect=True` only when the arm is already mechanically safe. The adapter
uses controller-planned `move_j()` and never uses the SDK's high-risk, unsmoothed `move_js()` path.
