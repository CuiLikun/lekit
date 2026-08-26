# Piper Cartesian and Isaac Teleop Design

**Date:** 2026-08-26

**Status:** Approved direction; implementation pending

## Objective

Extend `PiperRobot` with a standard LeRobot absolute TCP action and observation contract, then use a
`RobotProcessorPipeline` from `src/lekit/scripts/teleop.py` to retarget the cumulative engage-relative
pose produced by `IsaacXRController` into safe absolute Piper TCP targets at approximately 30 Hz.

## Constraints

- `IsaacXRController` remains a robot-independent input device. It must not import Piper code or read
  robot state.
- `PiperRobot.send_action()` remains a canonical robot command interface. It must not accept raw
  `left.*` or `right.*` Isaac fields.
- Public robot units are SI: metres, radians, seconds, and newtons.
- Cartesian commands address the configured TCP, not the flange. `pyAgxArm.move_p()` receives a flange
  pose, so `PiperRobot` performs the TCP-to-flange conversion internally.
- Joint, Cartesian, and gripper behavior already implemented by `PiperRobot` must remain compatible.
- Cartesian control is soft real-time target overwrite through controller-planned `move_p()`. It is not
  safety-rated Servo control and must never use `move_js()`.
- Automated tests must not connect to CAN or move physical hardware.

## Architecture

```text
IsaacXRController.get_action()
          |
          |  cumulative pose from this engage origin
          v
RobotProcessorPipeline[(RobotAction, RobotObservation), RobotAction]
  PiperIsaacRetargetingStep
    - select hand
    - validate tracking and pose
    - detect engage/release edges
    - latch measured TCP anchor
    - map operator frame to Piper base frame
    - compose translation and orientation
    - map trigger to gripper width
          |
          |  canonical absolute ee.* action
          v
PiperRobot.send_action()
    - reject mixed joint/Cartesian commands
    - validate finite complete TCP target
    - enforce final workspace and measured-pose lead limits
    - convert TCP target to flange target
    - call pyAgxArm.move_p()
```

`src/lekit/scripts/teleop.py` owns construction, connection order, the 30 Hz loop, Rich status output,
and best-effort hold/cleanup on release, tracking loss, exception, or interruption.

## Standard Piper action and observation contract

`PiperRobot.action_features` and `PiperRobot.observation_features` expose these Cartesian fields in
addition to the existing six joint fields and optional gripper:

```python
{
    "ee.x": float,
    "ee.y": float,
    "ee.z": float,
    "ee.roll": float,
    "ee.pitch": float,
    "ee.yaw": float,
}
```

The values describe the TCP pose in the Piper base frame. Translation is in metres. Orientation is
RPY in radians using the SDK convention `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.

Arm action families are mutually exclusive per call:

- one or more `joint_N.pos` fields select joint control;
- all six `ee.*` fields select Cartesian control;
- no arm fields select gripper-only or observation-only behavior;
- joint and `ee.*` fields in the same action are rejected;
- a partial Cartesian pose is rejected because filling it from hidden history makes replay unsafe;
- `gripper.pos` may accompany either complete arm representation.

`send_action()` returns one complete canonical action containing measured/accepted joints, the
measured/accepted TCP pose, and optional gripper width. The returned Cartesian values are the bounded
TCP target actually converted and sent to the SDK.

Before a Cartesian dispatch, `PiperRobot` also requires `arm.is_ok()` and valid measured joints/TCP
feedback. An unhealthy SDK/controller state rejects the complete action before `move_p()` or gripper
motion.

## Piper configuration

`PiperRobotConfig` gains:

```python
tcp_offset: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
eef_workspace_min_m: tuple[float, float, float] = (-0.65, -0.65, 0.02)
eef_workspace_max_m: tuple[float, float, float] = (0.65, 0.65, 0.75)
max_eef_target_lead_m: float | None = 0.005
max_eef_target_lead_rad: float | None = radians(2)
```

The TCP offset is a flange-frame `[x, y, z, roll, pitch, yaw]` transform. Zero is a valid TCP located
at the flange. Operators using the AGX gripper tip must configure the measured tool offset rather than
assuming the flange is the grasp point.

Workspace limits are a final rectangular guard in base coordinates, not a reachability proof. Lead
limits bound each outstanding target relative to fresh measured TCP feedback. Consequently, a stalled
application can leave at most a small configured motion outstanding, although a physical emergency
stop remains required.

Translation is clamped to the workspace and lead radius. Orientation is limited by shortest-path SO(3)
angular distance, never by independent Euler component clipping. The final safe rotation is converted
back to SDK RPY only at the SDK seam.

## Retargeting processor

`PiperIsaacRetargetingStep` is a registered, stateful LeRobot `RobotActionProcessorStep`. It accesses
the robot observation through `self.transition`, transforms only `TransitionKey.ACTION`, declares its
feature transformation, and implements `reset()`.

Its configuration includes:

- selected hand (`right` by default);
- translation and rotation gains;
- operator-to-base rotation;
- maximum displacement and rotation from one engage anchor;
- trigger-to-gripper mapping and width range;
- tolerances for the neutral first engage frame.

For the standard mounting preset, Piper base axes are treated as `+X forward`, `+Y left`, `+Z up`.
Isaac operator axes are `+X right`, `+Y forward`, `+Z up`. The preset therefore maps:

```text
operator right   -> Piper -Y
operator forward -> Piper +X
operator up      -> Piper +Z
```

The mapping is configurable and shown before motion is enabled. A dry run must be available so the
operator can verify directions without sending `move_p()`.

For an Isaac cumulative transform `(delta_p_operator, delta_R_operator)` and measured engage anchor
`(p_anchor, R_anchor)`:

```text
p_target = p_anchor + R_base_from_operator * scale * delta_p_operator

delta_R_base =
    R_base_from_operator * delta_R_operator * inverse(R_base_from_operator)

R_target = delta_R_base * R_anchor
```

Rotation is composed with matrices/quaternions. Euler angles are used only for the robot action
boundary.

## Engage state machine

```text
UNARMED --observe released+tracked--> IDLE
IDLE --neutral engage edge----------> ENGAGED
ENGAGED --valid engaged frame-------> ENGAGED
ENGAGED --release/tracking loss-----> IDLE (one measured hold command)
any state --invalid input-----------> FAULT (one measured hold command)
FAULT --observe released+tracked----> IDLE
```

- Starting the program while squeeze is already held does not arm motion. The operator must release
  and engage again.
- The engage edge latches the measured TCP from the same loop observation. The first engaged Isaac
  frame must be approximately zero translation and identity rotation.
- While held, Isaac values are cumulative from that engage origin; they are not integrated frame by
  frame.
- Release does not interpret Isaac's neutral output as a request to return to the anchor. It emits one
  fresh measured TCP hold command, clears the anchor, then emits empty actions while idle.
- Tracking loss follows the release path. Invalid shapes, non-finite values, malformed quaternions, or
  missing observation fields enter `FAULT` and require a release before rearming.
- Re-engagement always captures a new measured TCP anchor.

## Gripper mapping

The selected hand's trigger controls gripper width while engaged:

```text
trigger = 0 -> configured maximum opening
trigger = 1 -> configured minimum opening
```

The processor emits metres under `gripper.pos`. `PiperRobot` remains responsible for its final gripper
range and force limits. Idle frames omit gripper commands so release cannot unexpectedly change width.

## `teleop.py` behavior

The script uses dataclass/Draccus configuration in the same style as LeRobot scripts. It supports:

- Piper connection options and TCP offset;
- Isaac connection options;
- selected hand;
- 30 Hz default loop frequency;
- coordinate mapping and control gains;
- `enable_motion=False` by default;
- translation-only checkout by setting rotation gain to zero;
- bounded optional frame count for tests;
- Rich live status showing tracking, state, loop rate, measured TCP, target TCP, and fault reason.

Per loop:

```python
observation = robot.get_observation()
isaac_action = teleop.get_action()
piper_action = processor((isaac_action, observation))
if enable_motion:
    applied = robot.send_action(piper_action)
```

The loop uses latest-frame semantics and does not queue or replay missed frames. `precise_sleep()` paces
the loop. On normal interruption or an exception, the script attempts one measured hold command before
disconnecting the teleoperator and robot. Hard process termination and host failure remain outside this
software guarantee.

## Testing

Unit tests use fake SDK, robot, and teleoperator objects. They cover:

- TCP feature schemas, feedback parsing, TCP offset configuration, TCP-to-flange conversion, and
  `move_p()` calls;
- complete/mixed/partial action validation;
- workspace and measured-pose translation limits;
- SO(3) orientation limiting and Euler wrap behavior;
- processor feature transformation and reset;
- startup squeeze interlock;
- engage anchoring, cumulative translation, coordinate mapping, orientation composition, release hold,
  re-engagement, tracking loss, malformed input, and trigger mapping;
- `teleop.py` loop ordering, dry run, latest action dispatch, exception hold, and cleanup;
- existing Piper joint, gripper, camera, and demo tests.

## Hardware validation stages

Hardware validation is always separately approved and manually supervised:

1. Connect with motion disabled; verify live TCP and controller telemetry.
2. Keep rotation gain zero; engage without moving and verify no robot motion.
3. Test each operator translation axis separately with at most 10 mm hand displacement and low speed.
4. Release during a small translation and measure whether the hold target supersedes the in-flight
   `move_p()` target on firmware S-V1.8-8.
5. Test tracking loss and loop interruption with the same small displacement.
6. Enable one rotation axis at a time with at most 2 degrees command.
7. Verify re-engagement from a new hand position causes no TCP jump.
8. Test trigger/gripper only after the arm motion checks pass.

The operator keeps a physical emergency stop reachable and the workspace clear throughout.

## Non-goals

- Host-side inverse kinematics.
- `move_js()` or hard real-time servo control.
- Collision avoidance, singularity proof, or safety certification.
- Raw Isaac actions as canonical dataset actions.
- Automatic guessing of the physical TCP offset or robot-to-operator mounting transform.
