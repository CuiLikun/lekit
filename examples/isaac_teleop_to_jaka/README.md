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
- Targets must be sent continuously.
- The command period is step_num times 8 ms; step_num=4 pairs with
  dataset fps 30.
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
    --robot.control_mode=ee_pose \
    --robot.servo_step_num=4 \
    --robot.max_eef_step_m=0.01 \
    --robot.max_eef_step_rad=0.08 \
    --teleop.type=xr_controller \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=<hf_user>/<dataset_name> \
    --dataset.single_task="Pick up the object" \
    --dataset.fps=30 \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5
~~~

Hold the controller squeeze to engage motion. Releasing it holds the measured
TCP pose. The trigger is currently ignored because the JAKA driver does not
declare a gripper action.

Before running, verify the tool frame, user frame, payload, collision level,
workspace clearance, and emergency stop. The script powers on, enables, and
moves the physical robot.
