# JAKA Debugger

Start the live terminal debugger from the repository root:

```bash
uv run python -m robots.jaka_robot.debug_ui --ip 192.168.1.31
```

The default connection does not power on or enable the arm. Use the Power,
Enable, and Servo controls after verifying the workspace and emergency stop.

## Web Debugger

Start the browser debugger locally:

```bash
uv run python -m robots.jaka_robot.web_debug --ip 192.168.1.31
```

Open `http://<controller-host-ip>:8000` from any device on the same LAN; the
server listens on all local interfaces by default. State frames are delivered
through one WebSocket at 30 Hz by default, and relative +/- commands are
serialized with observation reads before reaching the robot SDK. Do not expose
physical robot controls on an untrusted network.

The web debugger selects motion automatically from the Servo button state.
With Servo On, a relative arm command is streamed for 0.2 seconds by default;
with Servo Off, it becomes one controller-planned joint or linear move. Adjust
the Servo target hold interval with `--relative-action-hold-s` when needed.

To enter an already checked setup with power and servos enabled:

```bash
uv run python -m robots.jaka_robot.debug_ui \
    --ip 192.168.1.31 \
    --power-on \
    --enable
```

The observation table refreshes at 30 Hz by default. Connection, power, enable,
and Servo buttons change label and color with their current state. Joint, TCP,
and gripper controls are shown beside their live values and limits; all groups
can be used without selecting a control mode. Each +/- click reads the latest
measured value and sends one bounded relative action.

JAKA SDK 1.7.2 does not expose model-specific joint or Cartesian position
bounds. When known, pass them explicitly; the debugger displays and enforces
the configured values:

```bash
uv run python -m robots.jaka_robot.debug_ui \
    --ip 192.168.1.31 \
    --joint-limits-json '{"joint_1.pos":[-3.14,3.14]}' \
    --eef-limits-json '{"ee.z":[0.10,0.80]}'
```
