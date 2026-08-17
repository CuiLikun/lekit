# SimRobot examples

A self-contained MuJoCo-backed robot arm that plugs into the LeRobot pipeline
through [`SimRobot`](/home/sorel/workspace/lekit/src/lekit/robots/sim_robot/sim_robot.py).
No physical hardware or external assets are required — the model, scripted
trajectory and dataset layout all live in this folder.

## Files

| File | Purpose |
| --- | --- |
| [`scene.xml`](./scene.xml) | Three-link MJCF arm with a fixed overhead camera and a wrist-mounted camera. `<position>` actuators give stable PD tracking. |
| [`record.py`](./record.py) | Drives `SimRobot` through scripted sine targets, writes `LeRobotDataset` episodes (joint state + 2 video MP4s + action). |
| [`replay.py`](./replay.py) | Loads the recorded dataset and sends each action back into the sim, summarising per-step tracking error. |
| [`visualize_live.py`](./visualize_live.py) | Live visualisation driver — opens the MuJoCo 3D viewer, OpenCV camera windows and/or rerun / foxglove streams, then runs a scripted policy. |

## Record + replay

```bash
MUJOCO_GL=egl PYTHONPATH=src python examples/sim_robot/record.py
MUJOCO_GL=egl PYTHONPATH=src python examples/sim_robot/replay.py
```

The replay reports per-step tracking error per episode — `lerobot.record` style
metrics, but produced by the same `SimRobot` that recorded the data.

## Live visualisation

`visualize_live.py` mirrors the `--display_data / --display_mode` flags from
[`lerobot.record`](https://huggingface.co/docs/lerobot/record) so the same
mental model works for the sim. Three orthogonal backends can be enabled
independently:

| Flag | Effect |
| --- | --- |
| `--display_mujoco_viewer` | Open the interactive MuJoCo 3D GUI (`mujoco.viewer.launch_passive`). Needs a display server (e.g. `MUJOCO_GL=glfw` on a desktop). |
| `--display_camera_windows` | Open one `cv2.namedWindow` per `SimCameraConfig`. Frame rate matches the live control loop. |
| `--display_lerobot_backend rerun \| foxglove` | Stream observation/action through `lerobot.utils.visualization_utils`. Rerun spawns a viewer (or connects to a remote `rerun --serve-web`); Foxglove exposes a WebSocket server the Foxglove app connects to. |

Examples:

```bash
# Desktop -- everything on, scripts policy in a loop until the 3D viewer is closed.
MUJOCO_GL=glfw PYTHONPATH=src python examples/sim_robot/visualize_live.py \
    --display_mujoco_viewer --display_camera_windows \
    --display_lerobot_backend rerun --steps 240

# Headless server -- only the rerun stream goes to a remote viewer.
# Start a rerun viewer on the host first:  rerun --port 9876
MUJOCO_GL=egl PYTHONPATH=src python examples/sim_robot/visualize_live.py \
    --display_lerobot_backend rerun \
    --display_lerobot_ip 127.0.0.1 --display_lerobot_port 9876 \
    --steps 200

# Headless server -- Foxglove (open the desktop app, point it at ws://localhost:8765).
MUJOCO_GL=egl PYTHONPATH=src python examples/sim_robot/visualize_live.py \
    --display_lerobot_backend foxglove --display_lerobot_port 8765 --steps 200
```

Backends degrade gracefully: on a headless host, the MuJoCo viewer logs a
GLFW / X-server error and continues without it. Rerun's first run will
log a winit error when no display server is reachable; rerun-on-a-server is
the supported headless path (`rerun --serve-web`).

### Wiring it into your own loop

```python
from lekit.robots.sim_robot import SimRobot, SimRobotConfig, SimRobotVisualizer

cfg = SimRobotConfig(
    xml_path="my_scene.xml",
    motors={...}, cameras={...},
    display_lerobot_backend="rerun",   # mirrored from lerobot.record
)
robot = SimRobot(cfg); robot.connect()
try:
    with SimRobotVisualizer(robot) as viz:        # auto-wires from cfg.display_*
        while viz.is_running:
            action = policy(robot.get_observation())
            sent = robot.send_action(action)
            viz.render(observation=robot.get_observation(), action=sent)
finally:
    robot.disconnect()
```

Explicit overrides (e.g. running the same `SimRobot` against two different
visualisers) accept the same kwargs as keyword arguments:

```python
viz = SimRobotVisualizer(
    robot,
    camera_windows=False,
    lerobot_backend="foxglove",
    lerobot_port=8765,
)
```

## Customising for your own MJCF

1. Replace `scene.xml` with your MJCF (any model with hinge/slide joints).
2. Edit the `motors={…}` and `cameras={…}` blocks in `record.py` /
   `visualize_live.py` so the keys match the `<joint name=…>` and
   `<camera name=…>` elements in your model.
3. Replace `scripted_action` with a real policy (or a `KeyboardTeleop`
   / `Phone` teleoperator when you want to drive the sim live).

The `calibration_dir` argument for `SimRobotConfig` is optional; without it,
the class falls back to a writable tmp location
(`$TMPDIR/lerobot/calibration/...`), which keeps sim artefacts out of the
user's `~/.cache/huggingface/lerobot` directory used by physical robots.
