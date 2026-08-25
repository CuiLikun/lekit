# Quest 3 Visualizer

Run the standalone real-device inspector:

```bash
uv run python -m lekit.teleoperators.isaac_teleop.quest3_visualizer
```

The program starts CloudXR when configured to do so, prints the workstation
address to enter in the Quest 3 CloudXR client, and opens Rerun Viewer. It has
no robot connection and never emits robot actions.

CloudXR may be ready before the headset connects. In that case the program
prints `Waiting for an XR headset to connect...` and keeps the runtime alive.
Open the client in Quest 3 and press Connect; startup continues as soon as the
headset becomes available. Use `--connect-timeout-s 60` when a bounded wait is
preferred. Error `-35` during this phase means OpenXR has no headset form
factor yet and is retried automatically.

Rerun uses two views. **Relative controller poses** displays both
squeeze-relative grip and aim transforms in a right/forward/up operator frame,
plus an independent trajectory for each current clutch hold. **Live status**
displays every `left.*` and `right.*` action field using the exact
`action_features` key names.

The views use fully-qualified Rerun selectors so that the panels include the
child entities under `/quest3`. If a Viewer was already open from an earlier
run, restart it or use its blueprint reset action after updating the module.

The visualizer verifies that a remote Rerun endpoint is reachable before
starting CloudXR. It then sends and flushes a neutral startup frame on a
relative `control_time` timeline. Therefore the 3D axes, neutral controller
poses, and `Waiting for the XR session` status are visible before Quest 3
connects. An unreachable endpoint fails immediately with `Cannot reach Rerun`
instead of leaving an apparently healthy process with an empty Viewer.

While waiting for the headset, the visualizer publishes a zero-valued
heartbeat once per OpenXR retry. `Published samples` in the Live status panel
must keep increasing. After the headset connects, the terminal prints a
`Quest frames: ...` line once per second. These two counters distinguish the
Rerun transport, XR connection, and controller-tracking stages.

The orange `/quest3` item under the **Blueprint** panel is a content-query root,
not a recorded entity. Selecting it can show `Entity not found in view`; this
does not mean the recording is empty. Recorded entities are under the
**Streams** panel at `/quest3/controllers` and `/quest3/status`.

Useful options:

```bash
# Give up if Quest 3 has not connected after 60 seconds.
uv run python -m lekit.teleoperators.isaac_teleop.quest3_visualizer \
    --connect-timeout-s 60

# Use an externally started CloudXR runtime for 10 seconds without opening a viewer.
uv run python -m lekit.teleoperators.isaac_teleop.quest3_visualizer \
    --no-auto-launch --no-viewer --duration-s 10
```

Press `Ctrl-C` to close the XR session cleanly.

## Remote SSH Viewer

Start the Rerun gRPC server on the local machine that has the display. This
command is intentionally headless and does not open a Viewer window:

```bash
rerun --serve-grpc --port 9876
```

Keep that terminal running. In a second local terminal, open the native Viewer
and connect it to the server:

```bash
rerun --connect rerun+http://127.0.0.1:9876/proxy
```

To use a browser instead, replace both local commands with:

```bash
rerun --serve-web --port 9876 --web-viewer-port 9090
```

Then open `http://127.0.0.1:9090` locally.

In another local terminal, create a reverse tunnel to the remote machine that
has Quest 3 and Isaac Teleop:

```bash
ssh -N -R 9876:127.0.0.1:9876 user@remote-host
```

Then run the visualizer on the remote machine. Its loopback Rerun endpoint is
carried back through the SSH tunnel to the local Viewer:

```bash
uv run python -m lekit.teleoperators.isaac_teleop.quest3_visualizer \
    --rerun-url rerun+http://127.0.0.1:9876/proxy
```

The Rerun address and the Quest 3 address are independent. `--rerun-url`
selects where visualization data is sent; in the Quest client, enter the LAN
address printed by the visualizer (for example `192.168.5.24`) so it connects
to CloudXR on the remote workstation.

## Server-hosted Web Viewer

When a native Viewer on the local computer is unreliable, let the visualizer
host both the Rerun proxy and Web Viewer on the SSH server:

```bash
uv run python -m lekit.teleoperators.isaac_teleop.quest3_visualizer \
    --serve-web
```

Forward both ports from a terminal on the local computer:

```bash
ssh -N \
    -L 9090:127.0.0.1:9090 \
    -L 9876:127.0.0.1:9876 \
    user@remote-host
```

Then open this URL locally:

```text
http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy
```

After restarting the server process, refresh this page so it subscribes to the
new in-memory recording. A current recording uses a relative `control_time`
such as `+12.3 s`; a timeline showing an old wall-clock-like value is stale.

Using loopback through SSH also gives the browser a secure-context exception
for WebGPU and avoids LAN firewall rules on both Rerun ports.
