# JAKA Python SDK and LeRobot `Robot` feasibility

## Verdict

Implementing a six-axis JAKA arm as a LeRobot-compatible `Robot` is feasible.
The SDK supplies the lifecycle, feedback, and command primitives required by
LeRobot. For policy inference and teleoperation, use JAKA Servo Move
(`servo_move_enable`, then `servo_j` or `servo_p`) rather than issuing a
planner-backed `joint_move` for every frame.

## Contract mapping

| LeRobot `Robot` responsibility | JAKA SDK capability | Implementation decision |
| --- | --- | --- |
| `connect()` / `disconnect()` | `RC(ip)`, `login()`, `logout()`, `power_on()`, `enable_robot()`, `disable_robot()` | Make power-on and enable explicit config flags; only disable on teardown when this process owns the enable state. |
| `is_connected` | No documented connection-state getter | Track a successful login locally and clear it after logout; turn non-zero SDK return codes into driver exceptions. |
| `get_observation()` | `get_actual_joint_position()` and `get_actual_tcp_position()` | Emit six joint fields in radians and optional TCP fields. Convert TCP XYZ from SDK millimetres to LeRobot metres. |
| `send_action()` for discrete moves | `joint_move()` / `linear_move()` | Suitable for homing and slow, one-off moves, but not a policy control loop. |
| `send_action()` for continuous control | `servo_move_enable()`, `servo_j()`, `servo_p()` | Enter Servo Move before streaming, submit absolute targets continuously, observe queue depth, and leave the mode in `disconnect()`. |
| gripper | Cabinet/tool/extension digital and analog IO APIs | Keep gripper configuration optional and hardware-specific. A generic JAKA arm driver can expose normalized `gripper.pos` only when its IO mapping and electrical range are configured. |

## Timing and safety constraints

- The JAKA document says Servo Move runs on an 8 ms controller cycle and recommends continuously sending commands at 8 ms. `step_num` multiplies that period. A LeRobot 30 Hz loop aligns approximately with `step_num=4` (32 ms), but the resulting cadence and jitter need measurement on the target controller/network.
- Servo Move bypasses the controller motion planner. The document explicitly requires the client to plan or filter its trajectory; otherwise motion can be severely jerky. The driver must therefore clamp per-frame joint or Cartesian deltas and reject non-finite inputs before calling the SDK.
- The documentation gives a 180 degree/s joint-speed limit for `servo_j`; a command above it is ineffective. Per-step bounds must be derived from the actual loop period, not only a static maximum delta.
- `servo_j` / `servo_p` return queue depth, whose maximum is documented as 100. The driver should warn or stop before the queue fills and treat `-62` (servo queue full) and `-63` (servo mode not started) as actionable failures.
- The SDK documentation uses millimetres and radians. LeRobot-facing Cartesian actions/observations should use metres and radians, with conversion at the driver boundary.
- Initial deployment must validate tool/user frame, payload, collision settings, workspace limits, and emergency stop on hardware. These are not guarantees provided by the generic LeRobot interface.

## Compatibility findings in this repository

- The installed `lerobot` `Robot` base requires feature schemas, connection and calibration state, lifecycle/configuration methods, `get_observation()`, `send_action()`, and `disconnect()`. JAKA covers the hardware-facing parts of that contract.
- The local SDK binary imports under this project's Python 3.12 when both `jkrc.so` and `libjakaAPI.so` are on the dynamic linker path. The JAKA 1.7.2 page states Linux Python 3.5+ (32-bit) and Windows Python 3.7+ (64-bit), so target deployment still needs an explicit import and hardware smoke test.
- The official page and local binary expose `login()` / `logout()`. The local `jkrc.pyi` instead declares `log_in()` / `log_out()`, so it is not a reliable API source. Use the tested runtime names, or a narrow compatibility shim that checks which spelling exists.

## Recommended first implementation scope

1. Build a `JakaRobotConfig` with controller IP, auto power/enable controls, a bounded control period, and optional camera/gripper configuration; infer joint or Cartesian dispatch from each action's fields.
2. Implement `JakaRobot` with fixed six-joint feature keys (`joint_1.pos` through `joint_6.pos`), optional `ee.*` and camera observations, centralized tuple-return/error handling, and no-op calibration unless a model-specific requirement is identified.
3. Implement Servo Move as the continuous action backend, including stateful enable/disable cleanup, target-rate limiting, queue monitoring, and tests using a fake `jkrc.RC` object.
4. Add a guarded hardware acceptance test: connect without automatic movement, verify feedback units/order, then execute a slow, bounded Servo Move trajectory while recording SDK latency, achieved cadence, and queue depth.

## Sources

- JAKA, [Python SDK 1.7.2](https://www.jaka.com/docs/guide/1.7.2/SDK/Python.html), accessed 2026-08-05. This documents environment requirements, units, lifecycle methods, feedback APIs, motion commands, Servo Move timing, queue limit, and error codes.
- Local JAKA SDK package: `src/lekit/robots/jaka_robot/jkrc.so`, `libjakaAPI.so`, and `jkrc.pyi`, inspected 2026-08-05. Runtime symbol inspection confirmed `login`, `logout`, Servo Move, joint/TCP feedback, status, and IO calls.
- Local LeRobot 0.6.0 installation: `lerobot.robots.robot.Robot`, inspected 2026-08-05. This defines the required standard robot contract.
