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
through WebSockets at 30 Hz by default. One shared observation loop reads the
robot once per frame regardless of how many browsers are open. Relative +/-
commands are serialized before reaching the robot SDK. Do not expose physical
robot controls on an untrusted network.

The web debugger selects motion automatically from the Servo button state.
With Servo On, a relative arm command updates the target owned by
`JakaRobot`; its internal sender plans bounded joint or Cartesian setpoints and
feeds the controller continuously every 8 ms. With Servo Off, the click becomes
one controller-planned joint or linear move. The Console reports the actual
Servo frequency, timing jitter, overruns, queue depth, target age, and watchdog
errors.

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

The Isaac XR recorder enables `separate_feedback_connection` automatically and
delays Servo Move startup until XR tracking is ready. This is important for
continuous Servo P control: JAKA `servo_p` occupies the control SDK handle for
almost the full 8 ms cycle, so synchronous feedback reads on that same handle
can starve the command stream.

JAKA SDK 1.7.2 does not expose model-specific joint or Cartesian position
bounds. When known, pass them explicitly; the debugger displays and enforces
the configured values:

```bash
uv run python -m robots.jaka_robot.debug_ui \
    --ip 192.168.1.31 \
    --joint-limits-json '{"joint_1.pos":[-3.14,3.14]}' \
    --eef-limits-json '{"ee.z":[0.10,0.80]}'
```

## Isolated `servo_p` Diagnostic

The raw diagnostic bypasses `JakaRobot`'s managed Servo thread and calls only
`servo_move_enable()` plus `servo_p(..., ABS, 1)` at cumulative 8 ms
deadlines. Start with a stationary hold; this sends the measured TCP pose
without requesting any movement:

```bash
uv run python -m robots.jaka_robot.servo_p_debug \
    --ip 192.168.1.31 \
    --mode hold \
    --duration-s 5 \
    --csv artifacts/servo_p_hold.csv
```

If the hold is stable, run the precomputed 1 mm minimum-jerk Z bump:

```bash
uv run python -m robots.jaka_robot.servo_p_debug \
    --ip 192.168.1.31 \
    --mode z-bump \
    --duration-s 4 \
    --settle-s 1 \
    --amplitude-mm 1 \
    --csv artifacts/servo_p_z_bump.csv
```

When a fixed-target hold still shakes, repeat the same hold with only JAKA's
Cartesian nonlinear filter changed:

```bash
uv run python -m robots.jaka_robot.servo_p_debug \
    --ip 192.168.1.31 \
    --mode hold \
    --duration-s 8 \
    --cartesian-nlf \
    --csv artifacts/servo_p_hold_nlf.csv
```

The diagnostic checks the filter call's SDK return code and restores no-filter
mode during cleanup.

When `servo_filter_mode="cartesian_nlf"`, the managed sender forwards the latest
Cartesian target directly at 8 ms intervals. The controller-side NLF is the only
velocity/acceleration/jerk trajectory filter; applying the same profile on the
host would double the lag. JAKA documents Cartesian NLF translation in mm units,
orientation in degrees, and recommends keeping linear jerk below 5000 mm/s^3 to
avoid persistent motion around a fixed target.

Add `--power-on --enable` only when the controller is not already ready. By
default, states changed by the script are restored after the test. The script
always exits Servo Move in `finally`, aborts when queue depth reaches 80 or
five timing periods overrun consecutively, and writes CSV only after the
real-time loop has stopped.

- A shaking hold points to SDK/network/controller timing rather than trajectory
  interpolation.
- A stable hold but shaking bump points to trajectory, units, frames, or robot
  dynamics.
- A send rate far below 125 Hz, large period p95/max, or rising queue depth
  identifies host scheduling or communication as the primary problem.
