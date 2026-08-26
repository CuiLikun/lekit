# Independent Isaac Teleop Node Design

## Goal

Run the existing `IsaacXRController` as the sole owner of CloudXR/OpenXR in a long-lived background process. Publish complete dual-controller snapshots over a transport boundary so any control or visualization process can subscribe without opening another XR session. Provide a read-only browser monitoring page in the same service.

## Scope

- The first transport is ZeroMQ PUB/SUB because Lekit already depends on `pyzmq`.
- The transport boundary remains replaceable so a Dora backend can be added later.
- The node never connects to or commands a robot.
- The node publishes one atomic action frame containing all 28 action keys.
- A standard LeRobot teleoperator subscriber exposes the remote stream through `connect()`, `get_action()`, and `disconnect()`.
- The monitoring page exposes service and controller state only. It has no motion-enable or robot-control endpoint.

## Module boundaries

- `protocol.py` owns the versioned wire schema, action normalization, neutral action, and shared action feature declaration.
- `transport.py` owns ZeroMQ publication and latest-frame subscription. It knows nothing about XR or robots.
- `node_state.py` owns the thread-safe monitoring snapshot and FastAPI application.
- `node.py` owns the hardware sampling lifecycle, service state, frame sequence, publication rate, reconnect loop, and CLI.
- `subscriber.py` is the standard LeRobot teleoperator adapter used by downstream control loops.

`IsaacXRController` remains the hardware adapter. Relative-pose clutching and head-yaw anchoring stay there and are never reconstructed by subscribers.

## Wire protocol

The action topic is `isaac_teleop/action/v1`. Each update is one complete
ZeroMQ message containing the topic bytes, one ASCII space, and one UTF-8 JSON
payload. A single-part message is intentional: ZeroMQ `CONFLATE` keeps the
latest complete message and does not support multipart messages. The payload
contains:

- `schema`: `lekit.isaac_teleop.action`
- `schema_version`: `1`
- `session_id`: a UUID generated when the node starts
- `sequence`: a non-negative integer increasing once per sampled XR frame
- `captured_monotonic_ns`: producer monotonic sampling time
- `captured_utc_ns`: producer wall-clock time for logs
- `action`: all 28 left/right action fields in one object

NumPy arrays are encoded as JSON arrays and reconstructed as `float32`. Scalars are finite floats or booleans. Decoding rejects missing, extra, malformed, non-finite, or incompatible fields.

The status topic is `isaac_teleop/status/v1`. It carries the same monitoring snapshot exposed by the Web API. Status is diagnostic and must not be interpreted as a robot command.

## Latest-frame and safety semantics

- PUB/SUB queues are bounded and use zero linger so a slow or absent consumer cannot block XR sampling or process shutdown.
- ZeroMQ `CONFLATE` retains only the newest complete action message; the
  subscriber additionally rejects duplicate or out-of-order sequence numbers
  within the current session.
- `connect()` waits for a first valid frame up to a configurable timeout.
- `get_action()` never replays an expired frame. If local receive age exceeds `stale_after_s`, it returns a neutral action with both hands untracked, not aim-tracked, and disengaged.
- A new publisher session invalidates the previous cache. The subscriber remains clutch-inhibited until it observes both squeeze values at or below `rearm_squeeze_threshold`; this prevents a held squeeze from automatically re-engaging after node restart.
- Producer timestamps are diagnostics. The safety watchdog uses the subscriber's local monotonic receive time so it works across machines without synchronized clocks.

## Node lifecycle

1. Start the monitoring server and expose `starting`.
2. Bind the ZeroMQ publisher.
3. Create `IsaacXRController` and expose `waiting_for_headset` while `connect()` retries.
4. On XR connection, expose `streaming` and sample at the configured rate.
5. On an XR/runtime exception, expose `reconnecting`, publish the fault status, cleanly close the controller, wait `retry_delay_s`, and create a new controller session.
6. On SIGINT/SIGTERM, expose `stopping`, close controller and sockets with zero linger, then stop the Web server.

Each process start and each successfully retried XR session gets a new `session_id`; sequence restarts at zero for that session. This makes a CloudXR/OpenXR failure visible to every subscriber and forces release-to-rearm even when the node process itself survives.

## Monitoring page

The FastAPI service provides:

- `GET /`: self-contained monitoring dashboard.
- `GET /api/status`: JSON snapshot.
- `GET /healthz`: process liveness and current state.

The page polls `/api/status` and displays node state, uptime, publish endpoint, Web endpoint, session, sequence, frame count, measured publish rate, last-frame age, last error, and every left/right action field. The page remains usable while the node is waiting for Quest or reconnecting.

The default Web bind is `127.0.0.1`; remote SSH users access it with port forwarding. Binding to `0.0.0.0` is an explicit CLI choice. No authentication is added in this first local-network version, so exposing it to an untrusted network is unsupported.

## Testing

- Protocol tests cover exact schema, round-trip types, invalid payloads, and neutral safety values.
- Real ZeroMQ tests cover late subscription, latest-frame conflation, and clean closure.
- Subscriber tests cover first-frame timeout, stale neutralization, session restart re-arming, and sequence rejection.
- Monitoring tests exercise the real FastAPI routes and verify complete action visibility.
- Node runner tests use a finite fake hardware source and in-memory publisher to verify state transitions and atomic frames without requiring Quest or a robot.

## Non-goals

- Robot kinematics, TCP mapping, gripper policy, or robot motion.
- Authentication, TLS, public-Internet exposure, or service discovery.
- Historical action storage or replay.
- Dora coordinator/daemon integration in the first implementation.
