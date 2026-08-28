# Piper Cartesian Servo Design

## Status

Approved behavior: prioritize continuous, bounded robot motion over exact instantaneous orientation tracking near singular configurations. Translation and gripper control remain responsive while orientation may temporarily lag and then catch up smoothly.

## Context

The current Quest-to-Piper path is:

1. Isaac Teleop publishes an engage-relative controller pose.
2. `PiperIsaacRetargetingStep` anchors it to the measured robot TCP and produces an absolute TCP target.
3. `PiperRobot.send_action()` bounds the target and sends it to `pyAgxArm.move_p()`.
4. Piper firmware independently solves inverse kinematics for each overwritten Cartesian target.

This works away from singular configurations. Near a wrist singularity, the firmware may select substantially different J4/J6 decompositions for adjacent TCP targets. Small input changes then produce large joint changes and visible vibration. The temporary `wrist_singularity_guard_rad` prevents vibration by rejecting rotational commands, but it also rejects the entire TCP action and makes the robot appear unresponsive in a common J5 posture.

## Goals

- Preserve the standard LeRobot absolute TCP action interface.
- Keep Teleop and Hub independent of Piper kinematics and robot feedback.
- Make Cartesian teleoperation continuous and responsive through common wrist postures.
- Give translation priority when orientation becomes ill-conditioned.
- Bound joint velocity and acceleration without a hard J5-angle switch.
- Allow orientation to lag near a singularity and catch up smoothly afterward.
- Keep gripper commands independent of Cartesian solver behavior.
- Expose enough diagnostics to distinguish input lag, target lag, singularity attenuation, and robot faults.

## Non-goals

- Exact instantaneous six-axis tracking at a mathematical singularity.
- Automatic large elbow or wrist reconfiguration on Engage.
- Changing the Controller, Control Handle, Hub, or transport protocols.
- Generalizing the first implementation into a kinematics framework for every robot.
- MIT motor pass-through control.

## Considered Approaches

### Disable the singularity guard and continue using `move_p()`

This restores all Cartesian commands but returns control of IK continuity to firmware. It already produced severe pitch vibration in real testing, so it is rejected.

### Keep `move_p()` and attenuate orientation near J5 zero

This is a useful fallback because translation can continue, but a J5 threshold is only a proxy for Jacobian conditioning and creates an arbitrary transition. It cannot prevent the firmware from changing IK branches. It is rejected as the primary solution.

### Robot-side differential Cartesian servo

This is the selected approach. The Robot computes small, continuous joint targets from measured joint feedback and an absolute TCP target. It uses adaptive damped least squares and sends bounded joint-position commands through `move_j()`, whose position-velocity mode retains lower-layer smoothing.

## Architecture

### External seam

The existing Robot interface remains unchanged:

```python
robot.send_action(
    {
        "ee.x": x,
        "ee.y": y,
        "ee.z": z,
        "ee.roll": roll,
        "ee.pitch": pitch,
        "ee.yaw": yaw,
        "gripper.pos": width,
    }
)
```

Callers do not know whether Piper uses firmware Cartesian IK or local differential IK. Hub and Controller behavior remains unchanged.

### Internal module

Add a deep `PiperCartesianServo` module under `src/lekit/robots/piper/`. Its public interface is intentionally small:

```python
step = servo.step(
    measured_joints=q,
    measured_tcp=current_pose,
    target_tcp=target_pose,
    dt=dt,
)
```

The returned value contains the next bounded joint-position target and diagnostics. Construction supplies model joint limits and an offline `fk_tcp(joints)` callable. The production adapter composes SDK `fk()` with `get_flange2tcp_pose()`. Tests supply deterministic kinematic adapters through the same seam.

The module owns:

- numerical Jacobian calculation;
- task-priority differential IK;
- adaptive singularity damping;
- joint velocity and acceleration limiting;
- soft joint-limit avoidance;
- previous command state for continuity;
- singularity and tracking diagnostics.

It does not read CAN, call `move_j()`, control the gripper, or manage Hub authority.

### Robot integration

`PiperRobot._send_eef_action()` continues to validate health, feedback, workspace, target lead, and gripper bounds. It then:

1. Reads measured joints and TCP pose.
2. Bounds the absolute TCP target as today.
3. Calls `PiperCartesianServo.step()`.
4. Selects joint motion mode once through the existing motion-mode cache.
5. Sends the returned joint target using `arm.move_j()`.
6. Sends a gripper command independently when requested.

The current hard singularity exception is removed. A solver fault fails closed, but ordinary singularity attenuation is a normal operating state rather than a Robot fault.

The servo resets when the Robot connects, disconnects, changes action representation, or starts a fresh control session. A reset seeds its state from measured joints so the first command cannot jump.

## Control Law

### Pose error

Translation error is measured in the base frame. Orientation error uses the shortest rotation vector rather than subtracting Euler angles, avoiding wraparound discontinuities.

The requested task velocity is generated from bounded proportional tracking:

```text
v_position = clamp(k_position * position_error, max_tcp_velocity)
v_rotation = clamp(k_rotation * rotation_error, max_tcp_angular_velocity)
```

Because the target remains absolute, any orientation motion attenuated in one cycle remains as error and is caught up in later cycles.

### Numerical Jacobian

The servo computes a 6×6 geometric Jacobian from the SDK's offline FK model using central finite differences around measured joints. Each FK result is converted to TCP pose before differencing. Rotational columns use relative rotation vectors, not Euler-angle subtraction.

At 30 Hz this requires twelve offline FK evaluations per control cycle and no additional CAN round trips. The implementation must record solver duration and remain below 5 ms at the 95th percentile on the deployment machine.

### Translation priority

Translation is solved first with a damped pseudoinverse. Orientation is solved in the remaining task space. When full orientation tracking is incompatible with bounded motion near a singularity, rotational response is attenuated before translation.

This guarantees that moving the hand still moves the TCP instead of causing the whole action to be rejected.

### Adaptive damping

The servo evaluates Jacobian singular values each cycle. Damping rises continuously as the smallest singular value falls; there is no `J5 == threshold` branch. The same condition metric limits rotational task velocity.

Expected behavior:

- Well-conditioned pose: negligible damping and near-complete pose tracking.
- Approaching singularity: progressively slower orientation response.
- At singularity: bounded continuous joints, translation retained, deficient orientation direction temporarily lags.
- Leaving singularity or stopping the hand: accumulated absolute orientation error is reduced smoothly.

### Joint continuity and limits

Per-joint velocity and acceleration limits apply after the differential solve. Soft joint-limit weights increase before hard model limits. The final joint target is clamped to the configured model limits and quantized through the existing Piper wire representation.

The first implementation does not add a null-space posture bias. It therefore cannot move the elbow or wrist for reasons not directly requested by the TCP target.

## Configuration

Replace `wrist_singularity_guard_rad` in the LeHub Piper configuration with servo parameters grouped under Piper Robot configuration. Initial conservative defaults are:

- control rate: 30 Hz;
- position tracking gain: 4.0 s⁻¹;
- orientation tracking gain: 4.0 s⁻¹;
- maximum TCP velocity: 0.10 m/s;
- maximum TCP angular velocity: 0.50 rad/s;
- maximum joint velocity: 0.50 rad/s per joint, further bounded by model limits;
- maximum joint acceleration: 1.50 rad/s² per joint;
- maximum joint jerk: 8.0 rad/s³ per joint;
- numerical-Jacobian step: 1e-4 rad;
- characteristic length for Jacobian row scaling: 0.25 m;
- normalized singular-value transition: 0.03 to 0.12;
- orientation response scale: smooth transition from 0.15 to 1.0 across that interval;
- damping: smooth transition from 0.15 at the low threshold to 0.005 at the high threshold.

These are Robot defaults and are not exposed through Teleop or Hub. The focused real-arm session may tune only these numeric values without changing the selected control law.

## State and Fault Handling

- Missing or stale joint/TCP feedback: hold measured joints and report a feedback fault.
- Non-finite target or FK result: hold and report a solver fault.
- Target outside workspace: keep existing workspace bounding behavior.
- Joint limit reached: attenuate the incompatible task direction and report limit saturation.
- Singular Jacobian: continue with adaptive damping; do not fault.
- Tracking loss or Hand Over: preserve the existing authority behavior and hold.
- Fresh Take Over: seed from measured state and wait for the existing neutral Engage anchor.

## Observability

Robot status diagnostics add:

- Cartesian servo state;
- smallest Jacobian singular value;
- applied damping;
- orientation response scale;
- position and orientation tracking error;
- maximum commanded joint velocity;
- solver duration;
- joint-limit saturation state.

These values are diagnostic only and do not change Hub authority semantics.

## Testing

### Offline tests

- Normal configurations produce continuous joint targets that reduce pose error.
- Adjacent TCP targets cannot produce discontinuous joint targets.
- A synthetic wrist singularity produces bounded joint velocity without exceptions.
- Translation continues while a singular orientation direction is attenuated.
- Repeated steps catch up to a held orientation target after conditioning improves.
- Velocity, acceleration, joint, workspace, and finite-value limits hold.
- Rotation-vector error remains continuous across Euler wraparound.
- Reset seeds from measured state and produces no first-frame jump.

### Piper Robot integration tests

- EEF actions use `move_j()` with complete bounded joint targets.
- Motion mode changes once, not once per frame.
- Gripper commands remain independent of orientation attenuation.
- Feedback or solver faults hold without emitting a partial unsafe joint target.
- Joint actions retain their existing behavior.

### Focused real-arm validation

One combined validation session will cover:

1. XYZ and gripper responsiveness with rotation enabled.
2. Roll, yaw, and pitch away from a singularity.
3. Slow pitch motion through J5 near zero.
4. Combined translation and rotation through the same region.
5. Release/re-engage continuity.

Success means no violent joint reversal, no dropped whole-pose frames due to singularity, no visible first-frame jump, and acceptable hand-to-robot latency. Real-arm tests retain the existing workspace and emergency-stop precautions but avoid multiple redundant approval gates.

## Rollout

1. Implement and validate the servo entirely offline with deterministic kinematic adapters.
2. Integrate it behind `PiperRobot.send_action()` without changing callers.
3. Remove the hard J5 guard from the LeHub configuration.
4. Run the focused real-arm validation at conservative limits.
5. Tune only damping and motion caps from captured diagnostics; do not modify coordinate mapping or Hub authority during this phase.
