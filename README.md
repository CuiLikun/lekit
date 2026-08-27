# Lekit Control Hub

The Control Hub runs as three independent processes. The Hub schedules short-lived exclusive
control handles; real-time actions travel directly from the Controller to the Robot.

```bash
uv run lekit hub --advertise-host 192.168.5.24
uv run lekit teleop --hub-seed tcp://192.168.5.24:5560
uv run lekit robot --kind piper --hub-seed tcp://192.168.5.24:5560
```

The third command is read-only by default: it registers the Piper and reports observations, but
rejects take-over and does not enable motion.

## Piper hardware boundary

Only enable motion after the workspace is clear, the emergency stop is reachable, calibration is
current, and the read-only process reports healthy status. Start the Robot with motion capability
explicitly enabled:

```bash
uv run lekit robot --kind piper \
  --hub-seed tcp://192.168.5.24:5560 \
  --enable-motion
```

`--enable-motion` does not move the Robot by itself. Motion additionally requires an explicit Hub
assignment and a fresh take-over. A Hub restart, Handle expiry, revoke, hand-over, tracking loss, or
stale action enters HOLD and requires release-to-rearm plus another fresh take-over; motion never
auto-resumes.

Piper and retargeting settings can be supplied as JSON objects with `--robot-config` and
`--processor-config`, or overridden with LeRobot-style dotted arguments such as
`--robot.channel=can0`, `--robot.speed_percent=10`, and `--processor.rotation_scale=0.5`.
