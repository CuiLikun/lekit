# Live Piper–LeHub Integration Design

## Goal

Connect the current LeHub browser design to one real Piper on local `can0`, one Hub-managed Isaac Quest 3 teleop node, and two cameras owned by the Piper robot node. The operator opens one LeHub page, sees the live robot and controller state, watches both native 640×480 camera feeds, and takes control by pressing the Quest 3 Engage input.

The deployment uses the existing Hub, `RobotNode`, `ControllerNode`, Piper adapter, Isaac teleop node, ZeroMQ management path, and direct latest-only action path. It does not introduce another camera transport process.

## Confirmed hardware

- Piper arm on `can0`, 1 Mbit/s.
- `HAND_VIEW`: Intel RealSense D435 color camera, serial `351323020301`, 640×480 at 30 FPS.
- `HEAD_VIEW`: Aoni A30, stable V4L2 path `/dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO._LTD_aoni_webcam_A30_20191119001-video-index0`, 640×480 at 30 FPS.
- One Quest 3 controller supplied by the existing Isaac teleop node.

Device identity must use the RealSense serial and the Aoni `/dev/v4l/by-id` symlink. Numeric `/dev/videoN` paths are diagnostic information only and must not be persisted in the launch configuration.

## Architecture

```text
Quest 3 --XR--> teleop-node ==60 Hz direct ZeroMQ==> robot node --CAN--> Piper
                       |             Control Handle             |
                       +---------------- Hub --------------------+

Browser --HTTP/MJPEG-------------------------------> robot node cameras
Browser --HTTP/WebSocket---------------------------> Hub state/control
Browser --HTTP-------------------------------------> teleop monitor
```

The three paths remain separate:

1. The existing Hub management path owns discovery, registration, the default route, Control Handle lifecycle, observability, and operator actions.
2. The existing direct Controller-to-Robot action path carries only Handle-fenced Isaac action envelopes. Hub and browser are never in this path.
3. A new Robot-owned video path carries camera metadata and latest-frame MJPEG streams directly to the browser. It cannot issue robot commands and does not share queues with the control path.

## Modules and seams

### Robot video module

Add a generic Robot video module whose small interface accepts named image observations and exposes a browser-facing read-only application:

```python
video = RobotVideoServer(config)
video.start()
video.publish(observation, captured_monotonic_ns=...)
video.describe() -> tuple[CameraStreamDescriptor, ...]
video.stop()
```

The implementation owns:

- capacity-one latest-frame storage per camera;
- RGB/BGR normalization and JPEG encoding;
- one asynchronous MJPEG response per connected browser;
- frame sequence, capture time, receive time, age, rate, and error diagnostics;
- bounded shutdown and slow-client isolation.

Publishing an observation must never wait for JPEG encoding or a browser. Each camera replaces its previous pending frame. JPEG work occurs outside the robot control loop. A failed camera or failed browser stream is local to that stream and cannot fault the CAN action loop unless the underlying standard Robot itself reports a camera failure while reading observations.

The video HTTP interface is read-only:

```text
GET /health
GET /api/cameras
GET /api/cameras/{camera_name}/stream.mjpg
```

`/api/cameras` returns only configured public metadata and current stream health. There are no POST, PUT, PATCH, or DELETE routes.

### RobotNode integration

The generic `RobotNode` remains the owner of the standard LeRobot Robot. It gains an optional observation sink interface. After `get_observation()` succeeds, `run_cycle()` offers the observation and capture time to each sink. Sink failures are recorded as diagnostics and do not block or replace Robot HOLD/SAFETY decisions.

The Piper CLI composes `RobotVideoServer` as an observation sink when video is enabled. `PiperRobot` continues to create and connect the two LeRobot camera adapters, so each physical device has exactly one owner.

Robot registration advertises a public video base URL and camera descriptors in typed node presentation metadata. The Hub stores and publishes this metadata but does not connect to, probe, or relay the video streams.

### Presentation metadata

Add optional presentation metadata to node registration rather than encoding browser URLs in free-form diagnostics. The minimum model is:

```text
presentation:
  monitor_url: optional HTTP URL
  video_status_url: optional HTTP URL
  cameras:
    - name
      stream_url
      width
      height
      fps
```

Controllers use `monitor_url`; Robots use `cameras`. Runtime adapters carry this metadata as ordinary low-rate management data. Action transport remains unchanged.

Wildcard bind hosts must never be advertised. CLI construction resolves a separately supplied advertise host into reachable URLs.

## Camera configuration

The first deployment configures:

```text
HAND_VIEW
  driver: RealSense
  serial: 351323020301
  color: 640×480 @ 30 FPS

HEAD_VIEW
  driver: OpenCV/V4L2
  path: /dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO._LTD_aoni_webcam_A30_20191119001-video-index0
  color: 640×480 @ 30 FPS
```

Piper JSON configuration must deserialize registered LeRobot camera config records into actual `CameraConfig` instances before constructing `PiperRobot`. The configuration file is the authority for camera names and profiles; the frontend renders exactly the descriptors reported by the Robot and never provides a camera-count selector.

## Automatic default route

The Hub adds an opt-in `auto_route_single_pair` setting, enabled by the target launch configuration. During reconciliation, the Hub creates a desired assignment only when all of these conditions hold:

- exactly one online, administratively enabled Robot exists;
- exactly one online Controller exists;
- their action schema and control mode are compatible;
- neither endpoint belongs to a non-terminal Handle;
- the Robot is not in SAFETY or FAULT.

The route is created through the same `Hub.assign` interface and persistence transaction used by an operator assignment. It receives an audit actor such as `hub:auto-route`; there is no second Handle implementation.

The Hub does not automatically request take-over. It distributes an `ASSIGNED` Handle and waits for the Controller's Engage transition. If either endpoint disappears before activation, normal expiry/revoke and Robot HOLD behavior apply. A released Handle may be recreated only after both endpoints are again compatible and idle. Reconciliation must be idempotent and must not generate duplicate Handles.

If multiple Robots or Controllers are online, automatic assignment is suspended and LeHub reports that manual routing is required.

## Engage-driven control

The Isaac teleop node observes the configured control hand's `is_engaged` value on every complete XR frame.

For each newly granted Handle:

1. The controller starts disarmed.
2. It must observe at least one valid frame with Engage released.
3. A subsequent released-to-pressed edge calls `controller.take_over(handle)`.
4. The existing Hub-routed take-over handshake arms the Robot and starts a fresh direct action stream.
5. A pressed-to-released edge calls `controller.hand_over(handle)` before further payload publication.

Repeated frames at the same Engage level are idempotent. Tracking loss, XR disconnect, node shutdown, Handle expiry, Hub revoke, or management loss clears the edge detector, disables publication, and requires a new released observation before another take-over. A Handle grant received while Engage is already pressed never starts motion.

The Piper processor retains its own clutching, anchor, workspace, gripper, stale-frame, and HOLD gates. Engage-driven Handle control adds an outer authority gate; it does not replace processor safety.

## LeHub frontend

Replace the old topology-first Hub page with the confirmed single-robot command-deck layout while preserving the existing HTTP and WebSocket control routes and hidden diagnostics/history views.

### Header

- One `LeHub` brand.
- Piper identity and current control state centered.
- One Quest 3 gamepad icon at the right.
- Controller icon state is derived from real node and Handle reports: offline, online, assigned, taking over, active, or fault.
- Clicking the icon opens the advertised teleop monitor URL. If no reachable URL is advertised, the click produces a notification rather than guessing an address.
- There is no duplicate gamepad button and no browser-simulated Engage control.

### Cameras

- Render exactly the camera descriptors advertised by the selected Robot.
- Show both `HAND_VIEW` and `HEAD_VIEW` at their native 640×480 dimensions.
- Do not scale the image element; allow natural horizontal layout/overflow when the browser cannot fit both feeds.
- Place the camera name at the lower-left image edge.
- Place a status dot and `LIVE`, `STALE`, or `OFFLINE` at the lower-right image edge.
- Stream state comes from successful image loading and Robot video diagnostics, not from a hard-coded label.
- The browser polls the advertised `video_status_url`; MJPEG image load events alone are not treated as proof that frames remain fresh.

### Runtime panel

Because no agent node is present, the page must not display simulated reasoning or task progress. The lower panel shows truthful live control information:

- topology: `Quest 3 → Retarget → Safety → Piper`;
- planning state: `Manual teleoperation · No agent assigned`;
- current Handle state, action frequency, frame age, processor state, tracking, Engage, and active HOLD.

The layout reserves the same presentation seam for a future agent task model, but this integration does not define or fabricate agent planning events.

### Faults and diagnostics

- New errors appear as transient toast notifications and remain in the notification center.
- Detailed node reports and Hub history stay in the hidden diagnostics panel.
- Robot, controller, or camera loss updates the visible state without reloading the page.
- Camera failure does not imply Robot FAULT unless the Robot node reports one; the affected camera card may independently show STALE or OFFLINE.

## Configuration and launch

The production CLI keeps the existing three user-facing node commands and adds only the configuration needed by their owned interfaces:

```text
lekit hub
  --auto-route-single-pair
  --web-host 0.0.0.0
  --advertise-host <LAN_HOST>

lekit teleop
  --hub-seed tcp://<LAN_HOST>:5560
  --action-endpoint tcp://0.0.0.0:5557
  --monitor-host 0.0.0.0
  --advertise-host <LAN_HOST>

lekit robot --kind piper
  --hub-seed tcp://<LAN_HOST>:5560
  --enable-motion
  --robot-config <piper-camera-config.json>
  --video-host 0.0.0.0
  --video-port 8081
  --advertise-host <LAN_HOST>
```

The exact option names may follow existing CLI conventions, but bind and advertise addresses must remain distinct. Startup output prints reachable Hub, monitor, and Robot video URLs.

## Failure and safety behavior

- No compatible unique pair: no automatic Handle and no motion.
- Engage held during startup/reconnect: remain disarmed until release then press.
- Hub management loss: action authority expires and Robot enters HOLD.
- Teleop/XR loss: Controller hands over when possible, stops publication, and Robot watchdog enters HOLD.
- Video server or browser loss: control continues; diagnostics report video degradation.
- Camera read failure surfaced by `PiperRobot`: Robot node reports the failure and follows its existing control-loop safety behavior.
- Robot video URL unreachable from the browser: the card shows OFFLINE and a notification; Hub never proxies as an implicit fallback.
- Multiple compatible nodes: automatic route is suspended; no arbitrary node is selected.

## Verification strategy

Automated verification uses fakes and local sockets only:

- camera config deserialization and stable device identity;
- capacity-one non-blocking observation sink and JPEG/MJPEG behavior;
- read-only video routes and per-camera health;
- registration presentation metadata encoding;
- idempotent unique-pair auto-routing and multi-node suppression;
- Engage release/press/release lifecycle, including startup-held and reconnect cases;
- Hub page rendering from real snapshots and camera descriptors;
- regressions for existing Handle fencing, HOLD, stale action, and Web routes.

Hardware verification is staged and separately authorized:

1. Read-only device and camera stream check with robot motion disabled.
2. Hub + teleop + robot registration and automatic `ASSIGNED` Handle check.
3. Engage edge check while motion remains disabled.
4. Explicitly approved Piper motion test with workspace clear and emergency stop reachable.

No automated test opens `can0`, connects the Quest 3, or sends Piper motion.

## Non-goals

- Agent task execution or synthetic agent reasoning.
- Hub-relayed video.
- WebRTC, recording, playback, or multi-view transcoding.
- Depth stream display from the D435.
- Multiple-Robot scheduling policy beyond suspending automatic routing.
- Replacing the existing ZeroMQ Runtime or direct action envelope.
