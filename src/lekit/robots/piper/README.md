# PiperRobot

`PiperRobot` adapts the AgileX `pyAgxArm` driver to LeRobot 0.6. Public units are SI: joint positions
and TCP RPY use radians, TCP translation and `gripper.pos` use metres. The six TCP fields
(`ee.x`, `ee.y`, `ee.z`, `ee.roll`, `ee.pitch`, `ee.yaw`) are measured TCP pose values in the Piper
base frame.

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
        firmware_version="v188",  # select the driver matching the controller firmware
        auto_enable=False,
        cameras={},
    )
)

with robot:
    observation = robot.get_observation()
```

TCP feedback fails closed unless the SDK's three end-pose component frames and synthesized TCP pose
all carry finite wall-clock timestamps within `tcp_feedback_max_age_s` (default `0.1` seconds). Initial
connection waits for complete, fresh TCP feedback before configuration or auto-enable; stale, missing,
or excessively future-dated feedback blocks commands instead of reusing an old pose.

Choose one action representation for a hold. Joint and TCP fields must never be mixed; a Cartesian
action must contain all six TCP fields. The values below all come from the same observation.

Joint hold (joint fields only):

```python
joint_hold = {
    "joint_1.pos": observation["joint_1.pos"],
    "joint_2.pos": observation["joint_2.pos"],
    "joint_3.pos": observation["joint_3.pos"],
    "joint_4.pos": observation["joint_4.pos"],
    "joint_5.pos": observation["joint_5.pos"],
    "joint_6.pos": observation["joint_6.pos"],
}
applied_joints = robot.send_action(joint_hold)
```

TCP hold (complete TCP fields only):

```python
tcp_hold = {
    "ee.x": observation["ee.x"],
    "ee.y": observation["ee.y"],
    "ee.z": observation["ee.z"],
    "ee.roll": observation["ee.roll"],
    "ee.pitch": observation["ee.pitch"],
    "ee.yaw": observation["ee.yaw"],
}
applied_tcp = robot.send_action(tcp_hold)
```

Do not construct a hold by iterating over the combined action schema: it exposes both action families
and would produce an invalid joint+TCP mixture.

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

The dynamic phase is blocked by static safety failures. The AGX feedback field `homing_status` is a
zeroing-status signal on the tested firmware, not a persistent "calibration complete" flag. A false
value is therefore reported as a warning but does not by itself skip the bounded gripper round trip;
health, fault bits, finite width feedback, and the configured width range remain hard gates. The demo
never calibrates the gripper or writes a zero point automatically.

## Hardware validation

The commands below were checked against the current Draccus `--help` output. They are operating
commands, not a claim of completed hardware validation. All physical checks require separate approval,
a supervisor, a reachable physical emergency stop, and a clear workspace.

### Isaac teleoperation dry run

Start the independent Isaac teleop node before the Piper process. This node is the sole owner of the
CloudXR/OpenXR runtime and may remain online across Piper sessions. Piper only subscribes to its
published control frames; it never starts, reconnects, or stops CloudXR. The validated fixed operator
frame uses `--no-head-yaw`:

```bash
uv run --extra teleop python -m lekit.teleoperators.isaac_teleop.teleop_node \
  --publish-endpoint tcp://127.0.0.1:5557 \
  --monitor-host 127.0.0.1 \
  --monitor-port 8000 \
  --no-head-yaw
```

`enable_motion` defaults to `false`. In this mode `teleoperate()` also forces `robot.auto_enable=False`,
even if a config file requests otherwise. The process subscribes to the node and reads robot feedback,
but the loop never calls `send_action()`:

```bash
uv run --extra piper --extra teleop python -m lekit.scripts.teleop \
  --robot.channel can0 \
  --robot.firmware_version v188 \
  --robot.auto_enable false \
  --teleop.endpoint tcp://127.0.0.1:5557 \
  --enable_motion false
```

Use this dry run first to verify the displayed hand, tracking state, and axis directions without
calling `move_p()`.

### Validated safe teleoperation defaults

`PiperIsaacTeleopConfig()` uses the conservative profile validated on the physical Piper checkout:

- motion remains disabled until `--enable_motion true` is explicit;
- robot speed is 10%, measured-position target lead is 5 mm, and measured-orientation target lead is
  1 degree;
- the subscriber uses `tcp://127.0.0.1:5557`, waits up to 5 seconds for its first frame, treats frames
  older than 0.25 seconds as stale, and requires both squeeze inputs below 0.3 before re-arming;
- operator-frame selection belongs to the independent node; starting it with `--no-head-yaw` uses the
  fixed frame validated for Piper;
- translation is limited to a 100 mm radius and rotation to 10 degrees from each Grip anchor;
- the AGX gripper is limited to 0–50 mm at 1 N;
- translation, rotation, and trigger mapping are enabled together.

After the dry run and bounded checkout have passed, start this profile with the measured TCP offset:

```bash
uv run --extra piper --extra teleop python -m lekit.scripts.teleop \
  --robot.channel can0 \
  --robot.firmware_version v188 \
  --robot.tcp_offset='[0, 0, 0, 0, 0, 0]' \
  --teleop.endpoint tcp://127.0.0.1:5557 \
  --enable_motion true
```

This command can move the arm and gripper. Keep the workspace clear and the physical emergency stop
reachable. Releasing Grip sends a fresh measured-pose hold; it does not disable the joints.

### Translation-only motion checkout

The following command has `enable_motion=true`: it really enables the arm and can move it. The six
`tcp_offset` entries are the `flange -> tool` transform `[x, y, z, roll, pitch, yaw]` in metres/radians.
Draccus accepts this tuple as one shell-quoted YAML/JSON-style list string; the expanded six-float
appearance in `--help` is misleading for actual parsing. The zero values are valid only when
measurement confirms that the configured TCP is at the flange; replace all six entries with the measured
offset for an attached tool.

Before the first controller frame is accepted, motion mode performs a read-only post-enable stability
check. By default, the measured TCP must remain within 1 mm and 0.5 degrees for a continuous 1 second
window; any larger settling motion restarts the window, and failure to stabilize within 10 seconds
aborts without entering the control loop. These limits can be adjusted with
`startup_stability_window_s`, `startup_stability_timeout_s`,
`startup_max_translation_drift_m`, and `startup_max_rotation_drift_rad`. The window and limits must be
positive, and the window must be shorter than the timeout; motion mode cannot bypass this gate.

```bash
uv run --extra piper --extra teleop python -m lekit.scripts.teleop \
  --robot.channel can0 \
  --robot.firmware_version v188 \
  --robot.tcp_offset='[0, 0, 0, 0, 0, 0]' \
  --robot.speed_percent 10 \
  --teleop.endpoint tcp://127.0.0.1:5557 \
  --processor.rotation_scale 0 \
  --enable_motion true
```

Run translation-only first. After it passes, repeat the same command with an explicit non-zero
`--processor.rotation_scale 1` and test one rotation axis at a time with no more than 2 degrees of
commanded rotation. Keep translation displacement to at most 10 mm during the first axis checks.

### Control and safety contract

- The default selected hand is the right hand. The standard preset maps operator right to Piper `-Y`,
  operator forward to Piper `+X`, and operator up to Piper `+Z`.
- Trigger `0` means maximum gripper opening; trigger `1` means minimum width (closed). Gripper commands
  are emitted only while the selected hand is engaged. Setting `robot.include_gripper=false`
  automatically makes teleoperation arm-only; `processor.include_gripper=false` also enables arm-only
  operation on a gripper-equipped robot, and trigger input is then ignored.
- Startup has a release interlock. If squeeze is already held when the program starts, it remains
  `UNARMED`; release it, then engage again from a neutral first frame. The neutral engage frame latches
  the measured TCP from that same observation and prevents a jump.
- A stale frame, publisher session change, sequence reset, or tracking loss is exposed as neutral,
  untracked input. The subscriber requires both squeeze inputs to be released before it accepts a new
  engagement, and the Piper processor emits a fresh measured-pose hold before clearing its anchor.
- Motion mode also has a pre-control TCP stability gate after motor enable. It sends no robot action
  while checking and resets the processor before the normal release interlock begins.
- Isaac poses are cumulative relative values from the engage origin, not frame-by-frame integration.
  The processor anchors them to the same-frame measured absolute TCP, then emits absolute TCP targets.
  Release or tracking loss emits one fresh measured TCP hold, clears the anchor, and requires release
  before re-engagement. Re-engagement captures a new measured anchor. Invalid input enters `FAULT` and
  also requires release before re-arming.
- There are two safety layers. The processor limits displacement/rotation from its engage anchor and
  validates the input; `PiperRobot` applies the final workspace, measured-pose lead, model joint-limit,
  orientation, and gripper limits before converting TCP to flange and calling controller-planned
  `move_p()`. Cartesian actions never use `move_js()`.
- Verify the mapping in dry run before motion. Keep the physical emergency stop reachable and the
  robot workspace clear for every enabled-motion test. Stop immediately on unexpected direction,
  tracking, feedback, or controller behavior.

Before sending a new joint target on a physical arm:

1. Confirm the CAN interface is up at 1 Mbps and the arm is mechanically supported.
2. Connect once with `auto_enable=False`, then verify `get_observation()` returns six plausible joint
   values, all six plausible TCP values, and a plausible gripper width.
3. Verify `robot.arm.get_joint_limits_enabled()` is `True`.
4. Use the separate six-joint hold or complete six-field TCP hold shown above; never combine the two
   representations. Confirm measured drift remains within the site's safety tolerance.
5. Only then test a small bounded target. Keep `max_relative_target` and the TCP workspace/lead limits
   enabled.

`disconnect()` intentionally leaves the joints enabled by default because disabling a raised Piper arm can
make it fall. Set `disable_on_disconnect=True` only when the arm is already mechanically safe. In
particular, disconnect is not an automatic disable or fall-prevention action.
