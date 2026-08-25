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

## Target robot adapters

An adapter owns all target-specific behavior. It consumes the relative action,
maps the operator frame into its target frame, establishes its target state
from its own feedback, and emits that robot's action schema. For JAKA this
means the adapter owns the operator-to-base transform, `ee.*` action
construction, and the JAKA Servo P lifecycle. None of these belong to this
module.
