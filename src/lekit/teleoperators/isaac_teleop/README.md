# Isaac XR Teleoperator

`lekit.teleoperators.isaac_teleop` is a standalone input module for a pair of
OpenXR controllers, including Quest 3 over CloudXR. It does not know about a
robot, robot base frame, TCP pose, inverse kinematics, Cartesian servo, or a
gripper implementation.

## Interface

`IsaacXRController.get_action()` returns device data in a stable operator
coordinate system:

```python
{
    "left.translation": np.ndarray,  # metres: (+right, +forward, +up)
    "left.rotation": np.ndarray,  # relative grip xyzw quaternion
    "left.aim_translation": np.ndarray,
    "left.aim_rotation": np.ndarray,  # relative aim xyzw quaternion
    "left.squeeze": float,
    "left.trigger": float,
    "left.thumbstick": np.ndarray,  # x, y
    "left.thumbstick_click": float,
    "left.primary_button": float,
    "left.secondary_button": float,
    "left.menu_button": float,
    "left.is_tracking": bool,
    "left.is_aim_tracking": bool,
    "left.is_engaged": bool,
    # The same fields are always present with the "right." prefix.
    "right.translation": np.ndarray,
    # ...
}
```

There is no hand-selection setting. One `get_action()` call advances the XR
session once and returns both controller snapshots from that frame. The flat,
side-prefixed keys match `action_features` and can be recorded directly by
LeRobot.

Each controller has an independent squeeze clutch owned by the teleoperator:

1. Before squeeze, `translation` is zero and `rotation` is identity.
2. On the squeeze engage edge, the current controller pose becomes the origin;
   the action remains neutral for that frame.
3. While held, `translation` and `rotation` are cumulative transforms from the
   engage pose, not per-frame increments.
4. On release or tracking loss, the action immediately becomes neutral. No
   robot pose is cached or restored.

With `use_head_yaw=true` (default), both controller frames are aligned to the
headset's horizontal heading only at squeeze engagement. The operator can turn
or re-position while released, while a head turn during a held gesture cannot
move a stationary controller.

The grip and aim poses have independent relative origins but share the same
controller squeeze engagement. Losing tracking resets only the affected hand.
Button and analog input decoding is independent of pose validity, so a
temporarily invalid grip pose does not erase otherwise valid button samples.

## Installation

```bash
uv sync --extra teleop
python -m isaacteleop.cloudxr --accept-eula
```

To inspect the input stream without a robot:

```bash
uv run python -m lekit.teleoperators.isaac_teleop.debug
```

`connect()` waits while OpenXR reports that no headset is available (`-35`).
Set `connect_timeout_s` in `IsaacTeleopConfig` to bound this wait; the default
is to wait until the operator connects or interrupts the process.

The module intentionally has no generic recording command: its action keys
are device data and do not match any robot's action schema by themselves.

## Independent teleop-node

`teleop_node` moves the XR session into one long-lived process. It publishes
the complete left/right action snapshot over ZeroMQ and hosts a read-only Web
monitor. It never creates or commands a robot.

Start it on the workstation connected to Quest 3:

```bash
uv run python -m lekit.teleoperators.isaac_teleop.teleop_node \
  --publish-endpoint tcp://127.0.0.1:5557 \
  --monitor-host 127.0.0.1 \
  --monitor-port 8000
```

Then open the NVIDIA Isaac Teleop client in Quest 3 and connect it to the
workstation as usual. The terminal reports the action endpoint and monitor
address. The monitor shows lifecycle state, frame rate and age, session and
sequence identifiers, the most recent error, and all 28 controller fields.
Tracking and input state are rendered on mirrored Quest 3 controller diagrams.
While a hand is engaged, a locally bundled Three.js scene animates its relative
grip pose, aim ray, and recent trajectory. The page does not require a CDN or
other public Web resource.

For an SSH-hosted workstation, forward the monitor port from your local
computer:

```bash
ssh -N -L 8000:127.0.0.1:8000 sorel@192.168.5.24
```

Open <http://127.0.0.1:8000> locally. Both the publisher and monitor bind only
to loopback by default. If subscribers must run on another trusted LAN
machine, bind the publisher explicitly:

```bash
uv run python -m lekit.teleoperators.isaac_teleop.teleop_node \
  --publish-endpoint tcp://0.0.0.0:5557 \
  --monitor-host 0.0.0.0
```

Remote subscribers must connect to the workstation's real address, for
example `tcp://192.168.5.24:5557`, never `tcp://0.0.0.0:5557`. This first
version has no transport authentication or encryption; expose it only on a
trusted network protected by the host firewall.

The service is designed to be supervised. For a quick background session:

```bash
tmux new-session -d -s isaac-teleop \
  'cd /home/sorel/workspace/lekit && uv run python -m lekit.teleoperators.isaac_teleop.teleop_node'
```

Use systemd or another process supervisor for unattended deployment. The node
itself reconnects after an XR runtime failure and gives every recovered XR
session a new `session_id`.

Each ZeroMQ action update is one complete message: the
`isaac_teleop/action/v1` topic, one ASCII space, then a versioned JSON payload.
This single-message layout allows ZeroMQ `CONFLATE` to retain the newest atomic
left/right snapshot. The monitor distinguishes XR samples, successfully
published frames, and frames dropped because the non-blocking transport was
under pressure.

### Subscribe from a control process

`IsaacTeleopNodeSubscriber` is a standard LeRobot `Teleoperator`. It connects
only to the ZeroMQ stream and never starts CloudXR/OpenXR:

```python
from lekit.teleoperators.isaac_teleop import (
    IsaacTeleopNodeConfig,
    IsaacTeleopNodeSubscriber,
)

teleop = IsaacTeleopNodeSubscriber(
    IsaacTeleopNodeConfig(
        endpoint="tcp://127.0.0.1:5557",
        first_frame_timeout_s=5.0,
        stale_after_s=0.25,
    )
)

with teleop:
    action = teleop.get_action()
```

The subscriber keeps only the latest complete frame. Missing or stale data
returns a complete neutral action with both hands untracked and disengaged.
After stale/invalid data or a publisher/XR session change, both squeeze inputs
must first be released before engaged actions are accepted again. This local
watchdog remains necessary even when a process supervisor or a future Dora
transport is used.

## Target robot adapters

An adapter owns all target-specific behavior. It consumes the relative action,
maps the operator frame into its target frame, establishes its target state
from its own feedback, and emits that robot's action schema. For JAKA this
means the adapter owns the operator-to-base transform, `ee.*` action
construction, and the JAKA Servo P lifecycle. None of these belong to this
module.
