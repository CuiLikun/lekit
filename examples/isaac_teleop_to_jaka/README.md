# Isaac Teleop to JAKA

This example records LeRobot datasets while an XR controller drives a JAKA
arm directly in Cartesian Servo Move mode. It does not build or invoke a
host-side IK solver. The controller receives absolute TCP targets through
servo_p.

The implementation follows the official JAKA Python SDK 1.7.2 documentation:
https://www.jaka.com/docs/guide/1.7.2/SDK/Python.html

- SDK Cartesian units are millimetres and radians; the LeRobot-facing driver
  exposes metres and radians.
- Servo Move must be enabled before servo_p and disabled during cleanup.
- `JakaRobot` continuously feeds interpolated targets to the SDK at the
  controller's 8 ms cycle. The XR loop only updates the desired target at the
  dataset rate.
- The controller reports a queue depth up to 100. The driver warns before the
  configured queue threshold and raises on SDK errors.
- The default XR transform targets the robot base frame, so this recorder
  requires user_frame_id=0.

Install the project and Isaac Teleop dependencies from the repository root:

~~~bash
uv pip install -e .
uv pip install "isaacteleop[cloudxr,retargeters-lite]~=1.3.131" "scipy>=1.14"
python -m isaacteleop.cloudxr --accept-eula
~~~

Run from the repository root:

~~~bash
python -m examples.isaac_teleop_to_jaka.record \
    --robot.type=jaka_robot \
    --robot.ip=192.168.1.31 \
    --robot.id=jaka_arm \
    --teleop.type=xr_controller \
    --robot.cameras="{ hand: {type: intelrealsense, serial_number_or_name: '342522070741', width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id="sorel/pick-cube" \
    --dataset.single_task="Pick up the object" \
    --dataset.fps=30 \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5
~~~

Hold the controller squeeze to engage motion. Releasing it holds the last
commanded TCP target, so a lagging feedback sample cannot pull the arm backward.
Each trigger press toggles the normalized `gripper.pos` command
between closed (`0`) and open (`1`).

The default operator mapping rebases OpenXR (X=Right, Y=Up, Z=Backward) into the JAKA
base frame (X=Forward, Y=Left, Z=Up): hand forward drives the robot forward, hand right
drives the robot right, hand up drives the robot up. A rotated operator station (CCW
yaw viewed from above) can be compensated with `--teleop.operator_yaw_deg`, or the
rebase can be overridden entirely with `--teleop.base_T_anchor`.

To control translation without rotating the tool, pass `--teleop.lock_pose=true`.
The current measured roll/pitch/yaw is captured when the clutch engages and held
while it remains engaged. `false` (the default) follows the XR controller's orientation.
Cartesian controller drift up to 0.2 mm is ignored by default; adjust it with
`--teleop.position_deadband_m` when the tracking noise or precision requirement differs.
The recorder also enables JAKA's `cartesian_nlf` automatically. Hardware hold
testing showed that unfiltered Servo P produced a fixed-target limit cycle,
while the same target was stable with the Cartesian nonlinear filter enabled.
The XR profile defaults to `0.15 m/s`, `0.8 m/s²`, and `8 m/s³` for linear Servo P
velocity, acceleration, and jerk to keep the control responsive while retaining
the nonlinear filter. Tune these with `--teleop.servo_linear_velocity_m_s`,
`--teleop.servo_linear_acceleration_m_s2`, and `--teleop.servo_linear_jerk_m_s3`.

To capture a synchronized control trace while reproducing motion or hold jitter, add:

~~~bash
    --control_trace_csv=artifacts/jaka_control_trace.csv
~~~

The CSV records raw OpenXR and transformed grip positions, whether each frame
came from XR or the disengaged hold latch, requested/applied/actual TCP poses,
target steps, tracking errors, and the managed Servo sender's internal target,
interpolated command, timing, overruns, and queue depth.

The driver always records `gripper.pos`. By default, gripper commands are sent
to JAKA extension analog output channel 3, with a width range of 0-1000 (0
closed, 1000 open), as expected by the referenced gripper controller. If your
wiring uses different channels or ranges, override the mapping explicitly. For
example, a cabinet AI0/AO0 mapped over 0-10 V:

~~~bash
    --robot.gripper_analog_input_enabled=true \
    --robot.gripper_analog_output_enabled=true \
    --robot.gripper_analog_input_iotype=0 \
    --robot.gripper_analog_input_index=0 \
    --robot.gripper_analog_output_iotype=0 \
    --robot.gripper_analog_output_index=0 \
    --robot.gripper_analog_input_min=0.0 \
    --robot.gripper_analog_input_max=10.0 \
    --robot.gripper_analog_output_min=0.0 \
    --robot.gripper_analog_output_max=10.0
~~~

JAKA SDK IO types are `0` for cabinet, `1` for tool, and `2` for extension IO.
Use the corresponding `*_inverted=true` option when the physical signal uses
the opposite open/closed direction. The SDK documentation does not define one
universal analog range, so use the gripper and controller electrical manuals
instead of assuming the example's 0-10 V range.

Before running, verify the tool frame, user frame, payload, collision level,
workspace clearance, and emergency stop. The script powers on, enables, and
moves the physical robot.
