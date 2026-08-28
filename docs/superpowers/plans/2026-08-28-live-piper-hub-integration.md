# Live Piper–LeHub Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the real LeHub page to one Hub-managed Piper, one Isaac Quest 3 teleop node, and two Robot-owned 640×480 camera streams, with automatic single-pair routing and Engage-driven authority.

**Architecture:** Keep Hub management, direct 60 Hz Controller-to-Robot actions, and Robot-to-browser video as three separate paths. Add typed presentation metadata to node registration, a generic non-blocking Robot observation/video seam, idempotent Hub auto-routing for one compatible pair, and an Engage edge gate in the Isaac teleop node.

**Tech Stack:** Python 3.12, LeRobot 0.6.0, FastAPI, Uvicorn, OpenCV, pyrealsense2/LeRobot RealSense camera, ZeroMQ, vanilla HTML/CSS/JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-live-piper-hub-integration-design.md`

## Global Constraints

- `HAND_VIEW` is RealSense D435 color, serial `351323020301`, 640×480 at 30 FPS.
- `HEAD_VIEW` is Aoni A30 through its stable `/dev/v4l/by-id/...-video-index0` path, 640×480 at 30 FPS.
- Robot node owns both camera devices and exposes read-only HTTP/MJPEG directly; no `camera-stream` process and no Hub video relay.
- Browser video clients and JPEG encoding must never block the Robot control loop.
- Hub remains outside the direct action path.
- Automatic routing runs only for exactly one compatible online Robot and one compatible online Controller.
- Engage must be observed released after every new grant/reconnect before a pressed edge may request take-over.
- Release, tracking loss, disconnect, expiry, revoke, or management loss must stop authority and preserve existing Robot HOLD/SAFETY behavior.
- No synthetic agent planning or progress may be shown when no agent node exists.
- Automated tests must not open `can0`, connect Quest 3, or move the Piper.
- Existing dirty-worktree changes belong to the user and must be preserved.
- Do not create Git commits unless the user separately authorizes commits; checkpoint steps below are verification-only.

---

## File structure

- `src/lekit/control/model.py` — typed camera and node presentation dataclasses carried by registration snapshots.
- `src/lekit/control/codec.py` — strict decoding of the new registration field.
- `src/lekit/control/controller.py` — controller presentation metadata and read-only current Handle access.
- `src/lekit/control/robot.py` — optional non-blocking observation sink seam and Robot presentation metadata.
- `src/lekit/control/video.py` — latest-frame JPEG stores, read-only MJPEG application, and bounded Uvicorn owner.
- `src/lekit/control/hub.py` — unique-pair auto-route reconciliation.
- `src/lekit/control/cli.py` — bind/advertise flags, camera config conversion, and process composition.
- `src/lekit/control/hub.html` — real single-Robot command deck bound to Hub snapshots and advertised URLs.
- `src/lekit/robots/piper/robot_node.py` — Piper camera/video composition without duplicating generic Robot behavior.
- `src/lekit/teleoperators/isaac_teleop/engage_authority.py` — per-Handle release/press/release state machine.
- `src/lekit/teleoperators/isaac_teleop/teleop_node.py` — invoke Engage state machine and advertise monitor URL.
- `configs/piper_lehub.json` — stable D435 and Aoni camera configuration.
- `src/lekit/robots/piper/README.md` — exact LAN startup and separately authorized hardware checks.
- Corresponding tests remain under `tests/control`, `tests/robots`, and `tests/teleoperators`.

---

### Task 1: Typed node presentation metadata

**Files:**
- Modify: `src/lekit/control/model.py`
- Modify: `src/lekit/control/codec.py`
- Modify: `src/lekit/control/controller.py`
- Modify: `src/lekit/control/robot.py`
- Modify: `src/lekit/control/__init__.py`
- Test: `tests/control/test_model_codec.py`
- Test: `tests/control/test_controller_node.py`
- Test: `tests/control/test_robot_node.py`

**Interfaces:**
- Produces: `CameraStreamDescriptor(name: str, stream_url: str, width: int, height: int, fps: float)`.
- Produces: `NodePresentation(monitor_url: str | None = None, video_status_url: str | None = None, cameras: tuple[CameraStreamDescriptor, ...] = ())`.
- Produces: `ControllerNodeConfig.presentation: NodePresentation` and `RobotNodeConfig.presentation: NodePresentation`.
- Produces: `ControllerNode.current_handle -> ControlHandle | None` as a frozen read-only authority snapshot.

- [ ] **Step 1: Write failing metadata validation and codec tests**

```python
def test_node_presentation_round_trip_is_typed_and_strict():
    presentation = NodePresentation(
        monitor_url="http://192.168.5.24:8000",
        video_status_url="http://192.168.5.24:8081/api/cameras",
        cameras=(CameraStreamDescriptor("HAND_VIEW", "http://192.168.5.24:8081/api/cameras/HAND_VIEW/stream.mjpg", 640, 480, 30.0),),
    )
    descriptor = controller_descriptor(presentation=presentation)
    decoded = decode_management(encode_management(registration(descriptor)))
    assert decoded.body["descriptor"]["presentation"]["cameras"][0]["name"] == "HAND_VIEW"

@pytest.mark.parametrize("url", ["", "tcp://192.168.5.24:8081", "http://0.0.0.0:8081/stream"])
def test_presentation_rejects_empty_non_http_and_wildcard_urls(url):
    with pytest.raises(ValueError):
        NodePresentation(monitor_url=url)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/control/test_model_codec.py -q`

Expected: import or constructor failure because `NodePresentation` and `CameraStreamDescriptor` do not exist.

- [ ] **Step 3: Add immutable model types and strict decode support**

Implement in `model.py`:

```python
@dataclass(frozen=True, slots=True)
class CameraStreamDescriptor:
    name: str
    stream_url: str
    width: int
    height: int
    fps: float

@dataclass(frozen=True, slots=True)
class NodePresentation:
    monitor_url: str | None = None
    video_status_url: str | None = None
    cameras: tuple[CameraStreamDescriptor, ...] = ()
```

Validate HTTP(S) URLs, reject wildcard advertised hosts, require positive dimensions/rate, and reject duplicate camera names. Add `presentation: NodePresentation = field(default_factory=NodePresentation)` to `NodeDescriptor`; update strict registration decode to reconstruct both dataclasses rather than leave nested dictionaries.

- [ ] **Step 4: Add config plumbing and current Handle access**

Add the presentation field to both node configs, pass it from `_descriptor_locked()`, and expose:

```python
@property
def current_handle(self) -> ControlHandle | None:
    with self._lock:
        return self._handle
```

Update existing descriptor fixtures to use the default empty presentation.

- [ ] **Step 5: Run focused and codec regression tests**

Run: `uv run pytest tests/control/test_model_codec.py tests/control/test_controller_node.py tests/control/test_robot_node.py -q`

Expected: all selected tests pass.

---

### Task 2: Non-blocking Robot video module

**Files:**
- Create: `src/lekit/control/video.py`
- Modify: `src/lekit/control/robot.py`
- Modify: `src/lekit/control/__init__.py`
- Create: `tests/control/test_video.py`
- Modify: `tests/control/test_robot_node.py`

**Interfaces:**
- Consumes: `CameraStreamDescriptor` and standard `RobotObservation` mappings.
- Produces: `ObservationSink.publish(observation: RobotObservation, *, captured_monotonic_ns: int) -> None`.
- Produces: `RobotVideoServer.start()`, `publish(...)`, `describe()`, and `stop()`.
- Produces: read-only `GET /health`, `GET /api/cameras`, and `GET /api/cameras/{name}/stream.mjpg`.

- [ ] **Step 1: Write failing capacity-one and read-only HTTP tests**

```python
def test_publish_replaces_pending_frame_without_waiting():
    entered, release = threading.Event(), threading.Event()
    def encoder(image):
        entered.set(); release.wait(1)
        return bytes([int(image[0, 0, 0])])
    store = LatestJpegStore(("HAND_VIEW",), encoder=encoder)
    store.publish({"HAND_VIEW": np.full((1, 1, 3), 1, np.uint8)}, captured_monotonic_ns=1)
    assert entered.wait(1)
    store.publish({"HAND_VIEW": np.full((1, 1, 3), 2, np.uint8)}, captured_monotonic_ns=2)
    store.publish({"HAND_VIEW": np.full((1, 1, 3), 3, np.uint8)}, captured_monotonic_ns=3)
    release.set()
    first = store.wait_encoded("HAND_VIEW", after_sequence=-1, timeout_s=1)
    latest = store.wait_encoded("HAND_VIEW", after_sequence=first.sequence, timeout_s=1)
    assert latest.jpeg == b"\x03"

def test_video_app_has_no_mutating_routes(video_app):
    client = TestClient(video_app)
    assert client.get("/api/cameras").status_code == 200
    assert client.post("/api/cameras").status_code == 405
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/control/test_video.py -q`

Expected: module import failure because `lekit.control.video` does not exist.

- [ ] **Step 3: Implement the capacity-one encoder store**

Use one `queue.Queue(maxsize=1)` and one daemon encoder worker per configured camera. `publish()` performs only validation and non-blocking replacement. Store immutable encoded records containing `sequence`, `captured_monotonic_ns`, `encoded_monotonic_ns`, `width`, `height`, and JPEG bytes. Default encoding is `cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])`; RGB observations are converted to BGR in the worker.

- [ ] **Step 4: Implement the read-only FastAPI/MJPEG adapter and bounded owner**

Build multipart frames as:

```python
b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
```

Each stream waits for a sequence newer than the one last sent and never accumulates a per-client queue. `RobotVideoServer.stop()` closes stores, signals Uvicorn, and joins within three seconds.

- [ ] **Step 5: Add the observation sink seam to RobotNode**

Define an `ObservationSink` Protocol and add `observation_sinks: tuple[ObservationSink, ...] = ()` to `RobotNode.__init__`. Immediately after a successful `robot.get_observation()`, call each sink with the same capture timestamp. Catch sink exceptions into `self._observation_sink_errors` and continue the current Robot control decision; never call a sink while holding the authority lock.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/control/test_video.py tests/control/test_robot_node.py -q`

Expected: all selected tests pass, including a test where a raising sink does not prevent a valid action from reaching `robot.send_action()`.

---

### Task 3: Piper camera construction and Robot-owned video CLI

**Files:**
- Modify: `src/lekit/control/cli.py`
- Modify: `src/lekit/robots/piper/robot_node.py`
- Modify: `src/lekit/robots/piper/__init__.py`
- Modify: `tests/control/test_cli.py`
- Modify: `tests/robots/test_piper_robot_node.py`
- Modify: `tests/robots/test_piper_robot.py`
- Create: `configs/piper_lehub.json`

**Interfaces:**
- Consumes: `RobotVideoServer`, `NodePresentation`, and LeRobot `OpenCVCameraConfig` / `RealSenseCameraConfig`.
- Produces: `_decode_piper_cameras(value: Mapping[str, object]) -> dict[str, CameraConfig]`.
- Produces CLI options `--video-host`, `--video-port`, and `--advertise-host` for `lekit robot`.

- [ ] **Step 1: Write failing camera deserialization and CLI tests**

```python
def test_piper_json_builds_registered_camera_configs():
    cameras = _decode_piper_cameras({
        "HAND_VIEW": {"type": "realsense", "serial_number_or_name": "351323020301", "width": 640, "height": 480, "fps": 30, "use_depth": False},
        "HEAD_VIEW": {"type": "opencv", "index_or_path": "/dev/v4l/by-id/aoni", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
    })
    assert isinstance(cameras["HAND_VIEW"], RealSenseCameraConfig)
    assert isinstance(cameras["HEAD_VIEW"], OpenCVCameraConfig)
    assert cameras["HEAD_VIEW"].index_or_path == Path("/dev/v4l/by-id/aoni")
```

Add parser assertions for a wildcard video host requiring `--advertise-host` at construction time and for default port `8081`.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/control/test_cli.py tests/robots/test_piper_robot_node.py -q`

Expected: missing decoder/arguments and dictionary camera configs rejected by `PiperRobot` construction.

- [ ] **Step 3: Implement strict camera config conversion**

Accept only `type == "realsense"` or `type == "opencv"`; reject unknown keys through constructor `TypeError` translated to `ValueError` naming the camera. Convert OpenCV string paths to `Path`. Import both config classes before `PiperRobotConfig` construction so LeRobot registration is available.

- [ ] **Step 4: Compose RobotVideoServer through an explicit Piper factory seam**

Add `observation_sinks: tuple[ObservationSink, ...] = ()` to `PiperNodeConfig` and pass it to `RobotNode(...)` in `make_piper_robot_node`. In `_build_piper_node`, derive camera descriptors from the decoded configuration, create a server only for configured RGB camera names, pass it as a sink before `node.start()`, advertise `http://<advertise-host>:<video-port>/...`, and stop the video server before closing the runtime. The empty tuple keeps the factory usable without HTTP in unit tests.

- [ ] **Step 5: Add the exact hardware config**

Create `configs/piper_lehub.json` with:

```json
{
  "channel": "can0",
  "cameras": {
    "HAND_VIEW": {
      "type": "realsense",
      "serial_number_or_name": "351323020301",
      "width": 640,
      "height": 480,
      "fps": 30,
      "use_rgb": true,
      "use_depth": false
    },
    "HEAD_VIEW": {
      "type": "opencv",
      "index_or_path": "/dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO._LTD_aoni_webcam_A30_20191119001-video-index0",
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    }
  }
}
```

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/control/test_cli.py tests/robots/test_piper_robot.py tests/robots/test_piper_robot_node.py tests/control/test_video.py -q`

Expected: all selected tests pass without opening real devices.

---

### Task 4: Engage-driven Controller authority and monitor advertisement

**Files:**
- Create: `src/lekit/teleoperators/isaac_teleop/engage_authority.py`
- Modify: `src/lekit/teleoperators/isaac_teleop/teleop_node.py`
- Modify: `src/lekit/teleoperators/isaac_teleop/__init__.py`
- Modify: `src/lekit/control/cli.py`
- Create: `tests/teleoperators/test_isaac_teleop_engage_authority.py`
- Modify: `tests/teleoperators/test_isaac_teleop_node.py`
- Modify: `tests/control/test_cli.py`

**Interfaces:**
- Consumes: `ControllerNode.current_handle`, `take_over(handle)`, `hand_over(handle)`, and normalized Isaac actions.
- Produces: `EngageAuthority(hand: str).update(action, controller) -> None` and `.reset(controller, *, release: bool) -> None`.
- Produces CLI `lekit teleop --advertise-host <LAN_HOST>` and advertised monitor URL.

- [ ] **Step 1: Write the failing state-machine tests**

```python
def test_new_handle_requires_release_before_takeover():
    gate, controller, handle = gate_fixture()
    controller.current_handle = handle
    gate.update(action(engaged=True, tracking=True), controller)
    assert controller.calls == []
    gate.update(action(engaged=False, tracking=True), controller)
    gate.update(action(engaged=True, tracking=True), controller)
    gate.update(action(engaged=True, tracking=True), controller)
    gate.update(action(engaged=False, tracking=True), controller)
    assert controller.calls == [("take_over", handle), ("hand_over", handle)]

def test_tracking_loss_hands_over_and_rearms():
    gate, controller, handle = gate_fixture()
    controller.current_handle = handle
    gate.update(action(engaged=False, tracking=True), controller)
    gate.update(action(engaged=True, tracking=True), controller)
    gate.update(action(engaged=True, tracking=False), controller)
    gate.update(action(engaged=True, tracking=False), controller)
    assert controller.calls == [("take_over", handle), ("hand_over", handle)]
    gate.update(action(engaged=True, tracking=True), controller)
    assert controller.calls == [("take_over", handle), ("hand_over", handle)]
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/teleoperators/test_isaac_teleop_engage_authority.py -q`

Expected: missing module/class failure.

- [ ] **Step 3: Implement the per-Handle edge reducer**

Track `(handle_id, armed_after_release, previous_engaged, controlling)` only. A Handle identity change clears all booleans. No Handle or invalid tracking calls `hand_over` once when controlling and resets. Catch `HandleNotGranted` / `HandleExpired` as benign authority races; propagate unexpected exceptions to the teleop reconnect/error path.

- [ ] **Step 4: Invoke the reducer before managed frame publication**

After `controller.get_action()` and before `_publish_frame(frame)`, call the gate for managed mode. On XR session finalization call `reset(control_node, release=True)`. Standalone teleop publishing behavior remains unchanged.

- [ ] **Step 5: Advertise the real monitor URL**

Construct `NodePresentation(monitor_url=f"http://{advertise_host}:{monitor_port}")` while the monitor binds to `monitor_host`. Reject advertising `0.0.0.0`; when monitor is disabled, advertise `None`.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/teleoperators/test_isaac_teleop_engage_authority.py tests/teleoperators/test_isaac_teleop_node.py tests/control/test_controller_node.py tests/control/test_cli.py -q`

Expected: all selected tests pass, including startup-held Engage, repeated level, release, tracking loss, and reconnect cases.

---

### Task 5: Idempotent Hub single-pair auto-routing

**Files:**
- Modify: `src/lekit/control/hub.py`
- Modify: `src/lekit/control/cli.py`
- Modify: `tests/control/test_hub.py`
- Modify: `tests/control/test_end_to_end.py`
- Modify: `tests/control/test_cli.py`

**Interfaces:**
- Consumes: existing `Hub.assign(robot, controller, control_mode="teleop", actor=...)` transaction.
- Produces: `HubConfig.auto_route_single_pair: bool = False`.
- Produces CLI `lekit hub --auto-route-single-pair`.

- [ ] **Step 1: Write failing reconciliation tests**

```python
def test_tick_auto_assigns_exactly_one_compatible_pair(hub_with_auto_route):
    register_robot_and_controller(hub_with_auto_route)
    hub_with_auto_route.tick()
    controls = hub_with_auto_route.get_snapshot().controls
    assert len(controls) == 1
    assert controls[0].handle_state is HandleState.ASSIGNED
    hub_with_auto_route.tick()
    assert len(hub_with_auto_route.get_snapshot().controls) == 1

def test_auto_route_suspends_for_multiple_robots_or_faulted_robot():
    hub = make_auto_route_hub()
    register(hub, robot_descriptor("piper-01"))
    register(hub, robot_descriptor("piper-02"))
    register(hub, controller_descriptor("quest3-main"))
    hub.tick()
    assert hub.get_snapshot().controls == ()

    single = make_auto_route_hub()
    robot = register(single, robot_descriptor("piper-01"))
    register(single, controller_descriptor("quest3-main"))
    single.receive_report(report(robot, runtime_state=RuntimeState.FAULT, error="camera timeout"))
    single.tick()
    assert single.get_snapshot().controls == ()
```

Also assert a released terminal Handle results in at most one fresh assignment once both reports are idle.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/control/test_hub.py -q`

Expected: `HubConfig` rejects or ignores `auto_route_single_pair` and no Handle appears.

- [ ] **Step 3: Implement `_reconcile_auto_route_locked()`**

Call it from `tick()` after liveness/expiry/mismatch handling and before publishing the snapshot. Filter current-session online nodes by role and schedulability. Return without mutation unless each candidate list has length one and no non-terminal Handle references either endpoint. Invoke the same internal assignment path as public `assign` with actor `hub:auto-route`; suppress only expected `NodeUnavailable`, `IncompatibleNode`, and `ControlConflict` races and emit an alert/audit entry for the refusal.

- [ ] **Step 4: Wire the opt-in CLI flag**

Default remains false for backwards compatibility. The real launch uses `--auto-route-single-pair`.

- [ ] **Step 5: Add a MemoryRuntime end-to-end authority test**

Start Hub, RobotNode, and ControllerNode; pump until the automatic Handle is granted. Feed gate actions `released → pressed`, pump until ACTIVE, publish one action, then feed released and assert `RELEASED` plus Robot HOLD. Use fake Robot and no real device.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/control/test_hub.py tests/control/test_end_to_end.py tests/control/test_cli.py -q`

Expected: all selected tests pass and repeated `tick()` calls never duplicate a live Handle.

---

### Task 6: Bind the confirmed LeHub command deck to real state

**Files:**
- Modify: `src/lekit/control/hub.html`
- Modify: `tests/control/test_web.py`

**Interfaces:**
- Consumes: existing `/api/snapshot`, `/api/history`, `/ws`, Handle control routes, and `NodeDescriptor.presentation`.
- Produces: native-size camera cards, one Quest controller icon, real control pipeline status, notifications, and hidden diagnostics.

- [ ] **Step 1: Replace old DOM-contract tests with failing command-deck tests**

```python
def test_root_exposes_single_robot_command_deck():
    html = client.get("/").text
    assert 'id="robot-identity"' in html
    assert 'id="quest-controller"' in html
    assert 'id="camera-stage"' in html
    assert 'id="runtime-panel"' in html
    assert 'id="log-panel"' in html
    assert "Create a control route" not in html
    assert "Simulate Engage" not in html
```

Add assertions for the exact strings `descriptor.presentation.cameras`, `presentation.video_status_url`, `presentation.monitor_url`, and `Manual teleoperation · No agent assigned`; also assert that `camera-count`, `Simulate Engage`, and generated SVG camera placeholders are absent.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/control/test_web.py -q`

Expected: old topology/routing markup violates the new DOM contract.

- [ ] **Step 3: Port the confirmed prototype layout into packaged `hub.html`**

Keep the LeRobot light palette, native `--feed-width:640px; --feed-height:480px`, lower-left name overlay, lower-right live status, top-centered Piper identity, and one right-side gamepad icon. Preserve notification, toast, diagnostics, history, and reduced-motion support from the real page.

- [ ] **Step 4: Render only real snapshot data**

Select the sole online Robot and Controller. Camera cards come only from `robot.descriptor.presentation.cameras`; set `<img src>` to each advertised stream URL. Poll `robot.descriptor.presentation.video_status_url` once per second and map each reported frame age/state to LIVE, STALE, or OFFLINE; `<img>` errors immediately mark that card OFFLINE. If cardinality is not one, show an explicit selection-required/ambiguity state and no guessed route.

Map the lower panel to `Quest 3 → Retarget → Safety → Piper`, `Manual teleoperation · No agent assigned`, Handle state, action rate, frame age, tracking, Engage, processor state, and active HOLD. Remove prototype-generated camera SVGs, local Engage toggles, and fabricated task percentages.

- [ ] **Step 5: Keep operational failure paths**

WebSocket reconnects with snapshot fallback. Alerts and action failures become toasts plus notification records. Controller icon click opens only a validated advertised monitor URL. Camera errors remain camera-local unless the Hub snapshot independently reports Robot FAULT/SAFETY.

- [ ] **Step 6: Run Web tests**

Run: `uv run pytest tests/control/test_web.py -q`

Expected: all Web facade and DOM-contract tests pass.

---

### Task 7: Full software integration and operator runbook

**Files:**
- Modify: `src/lekit/robots/piper/README.md`
- Modify: `README.md`
- Test: all files changed by Tasks 1–6

**Interfaces:**
- Consumes: final `lekit hub`, `lekit teleop`, and `lekit robot` CLI options.
- Produces: exact local/LAN startup sequence and explicit read-only versus motion-enabled gates.

- [ ] **Step 1: Document the exact startup sequence for host `192.168.5.24`**

```bash
uv run lekit hub \
  --management-endpoint tcp://0.0.0.0:5560 \
  --advertise-host 192.168.5.24 \
  --web-host 0.0.0.0 --web-port 8080 \
  --auto-route-single-pair

uv run lekit teleop \
  --hub-seed tcp://192.168.5.24:5560 \
  --action-endpoint tcp://0.0.0.0:5557 \
  --monitor-host 0.0.0.0 --monitor-port 8000 \
  --advertise-host 192.168.5.24

uv run lekit robot --kind piper \
  --hub-seed tcp://192.168.5.24:5560 \
  --robot-config configs/piper_lehub.json \
  --video-host 0.0.0.0 --video-port 8081 \
  --advertise-host 192.168.5.24
```

Document adding `--enable-motion` only for the separately approved motion phase.

- [ ] **Step 2: Run the full automated control/camera/teleop suite**

Run: `uv run pytest tests/control tests/cameras tests/robots/test_piper_robot.py tests/robots/test_piper_robot_node.py tests/teleoperators -q`

Expected: zero failures; no hardware device is opened by tests.

- [ ] **Step 3: Run static checks on changed Python files**

Run: `uv run ruff check src/lekit/control src/lekit/robots/piper src/lekit/teleoperators/isaac_teleop tests/control tests/robots/test_piper_robot.py tests/robots/test_piper_robot_node.py tests/teleoperators`

Expected: zero Ruff errors.

- [ ] **Step 4: Perform only the authorized read-only hardware check**

With `can0` already UP, start the three processes without `--enable-motion`. Verify:

```text
Hub shows exactly one Piper and one Quest 3 online.
One ASSIGNED default Handle exists.
HAND_VIEW and HEAD_VIEW each report 640×480 and update near 30 FPS.
Holding Engage before startup does not activate the Handle.
Release then press changes only Controller/Handle state because Robot motion is disabled.
```

Stop here and request explicit motion authorization before adding `--enable-motion`.

---

## Execution notes

- Run each focused test command immediately after its task; do not defer failures to Task 7.
- Use `apply_patch` for source edits and preserve unrelated dirty-worktree changes.
- Do not commit checkpoints without a separate user request. If commits are later authorized, create one commit per completed task using only that task's paths.
